#include "config.h"
#include "single_frame.h"

DEFINE_int32(experience_size, 100000, "experience_size");
DEFINE_int32(learning_steps_burnin, -1, "learning_steps_burnin");
DEFINE_int32(learning_steps_total, 500000, "learning_steps_total");
DEFINE_int32(epsilon_min, 0.1, "epsilon_min");
DEFINE_double(gamma, 0.95, "gamma");
DEFINE_int32(display_interval, 1, "display_interval");
DEFINE_int32(display_after, 10000, "display_after");

typedef std::array<float,num_actions> net_input_type;
typedef boost::shared_ptr<SingleFrame> SingleFrameSp;
typedef std::array<SingleFrameSp,temporal_window+1> InputFrames;

bool is_valid_action(int action) { return action >= 0 && action < num_actions; }
bool is_valid_reward(float reward) { return reward >= -1.0 && reward <= 1.0; }
bool is_valid_epsilon(float eps) { return eps >= 0.0 && eps <= 1.0; }
bool is_valid_q(float val) { return !std::isnan(val); }

struct Experience
{
	InputFrames input_frames;
	int action;
	float reward;
	SingleFrameSp next_frame;	

	void check_sanity() const
	{
		assert(is_valid_reward(reward));
		assert(is_valid_action(action));
	}
};

struct Policy
{
	int action;
	float val;

	Policy() {}
	Policy(void*) : action(-1), val(FLT_MIN) {}
	Policy(int action,float val = FLT_MIN) : action(action), val(val) 
	{
		assert(is_valid_action(action));
		assert(is_valid_q(val));
	}

	std::string to_string() const 
	{ 
		return val == FLT_MIN ? str(format("%d:rand")%action) : str(format("%d:%.2f")%action%val); 
	}
};

class AnnealedEpsilon
{
public:
	float epsilon, epsilon_min, epsilon_test_time;
	int age;
	bool is_learning;
	int learning_steps_total, learning_steps_burnin;
	mutable std::mt19937 random_engine;

	AnnealedEpsilon()
	: is_learning(true), age(0), epsilon_min(FLAGS_epsilon_min), learning_steps_burnin(FLAGS_learning_steps_burnin < 0 ? FLAGS_learning_steps_total / 10 : FLAGS_learning_steps_burnin), learning_steps_total(FLAGS_learning_steps_total), epsilon_test_time(0.1)
	{}

	float get() const
	{
		if (is_learning)
		{			
			return std::min(1.0f,std::max(epsilon_min, 1.0f-(float)(age - learning_steps_burnin)/(learning_steps_total - learning_steps_burnin)));
		}
		else
		{
			return epsilon_test_time;
		}
	}

	bool should_do_random_action() const
	{
		const float dice = std::uniform_real_distribution<float>(0,1)(random_engine);		
		const float eps = get();
		assert(is_valid_epsilon(eps));
		return dice < eps;
	}

	void operator ++()
	{		
		age++;
	}
};

class DeepNetwork
{
public :
	typedef std::array<float,MinibatchSize * InputDataSize> FramesLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> TargetLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> FilterLayerInputData;

	class ReplayMemory
	{
	public :
		DeepNetwork& net;

		ReplayMemory(DeepNetwork& net)
		: size(FLAGS_experience_size), net(net)
		{
			experiences.reserve(size);
		}

		bool has_enough() const
		{
			return experiences.size() > net.epsilon.learning_steps_burnin;
		}

		const Experience& get_random() const
		{
			return experiences[ net.randint(experiences.size()) ];
		}

		void push(const Experience& e)
		{
			++net.epsilon;

			Experience* out;

			// being slow for limited time frames
			if (experiences.size() < size)
			{
				experiences.push_back(e);				
			}
			else
			{
				experiences[net.randint(size)] = e;
			}
		}

	private:
		int size;
		std::vector<Experience> experiences; // reserved, no fragmentation
	};

	class Feeder
	{
	public :
		DeepNetwork& net;
		
		typedef shared_ptr<caffe::MemoryDataLayer<float>> MemoryDataLayerSp;

		MemoryDataLayerSp frames_input_layer;
		MemoryDataLayerSp target_input_layer;
		MemoryDataLayerSp filter_input_layer;
		TargetLayerInputData dummy_input_data;

		Feeder(DeepNetwork& net)
		: net(net)
		{			
		}

		void init()
		{
			cache_layers();
			check_sanity();

			std::fill(dummy_input_data.begin(),dummy_input_data.end(),0.0);
		}

		void input( const FramesLayerInputData& frames, const TargetLayerInputData& target, const FilterLayerInputData& filter )
		{
			frames_input_layer->Reset(const_cast<float*>(frames.data()),dummy_input_data.data(),MinibatchSize);
			target_input_layer->Reset(const_cast<float*>(target.data()),dummy_input_data.data(),MinibatchSize);
			filter_input_layer->Reset(const_cast<float*>(filter.data()),dummy_input_data.data(),MinibatchSize);
		}

		void input( const FramesLayerInputData& frames )
		{
			input( frames, dummy_input_data, dummy_input_data );
		}

		void cache_layers()
		{
			auto get_layer = [&](const char* name){
				auto result = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net.net->layer_by_name(name));
				assert(result);
				return result;
			};		

			frames_input_layer = get_layer("frames_input_layer");		
			target_input_layer = get_layer("target_input_layer");
			filter_input_layer = get_layer("filter_input_layer");
		}

		void check_sanity()
		{
			auto check_blob_size = [&](const char* blob_name,int n,int c,int h,int w) {
				auto blob = net.net->blob_by_name(blob_name);
				assert(blob);
				assert(blob->num() == n);
				assert(blob->channels() == c);
				assert(blob->height() == h);
				assert(blob->width() == w);
			};

			check_blob_size("frames",MinibatchSize,temporal_window+1,sight_diameter,sight_diameter);
			check_blob_size("target",MinibatchSize,num_actions,1,1);
			check_blob_size("filter",MinibatchSize,num_actions,1,1);
		}
	};

	template <int N>
	class Evaluator
	{
	public:
		static_assert(N <= MinibatchSize, "Invalid use of evalutor");

		DeepNetwork& net;
		Evaluator(DeepNetwork& net) : net(net) {}

		std::array<Policy,N> policies;
		std::array<Policy,N>& evaluate(const std::array<InputFrames,N>& input_frames_batch,std::function<bool(int)> is_valid_action)
		{			
			FramesLayerInputData frames_input;
			auto target = frames_input.begin();
			for (const auto& input_frames : input_frames_batch)
			{
				SingleFrame::fill_frames(target,input_frames);
			}  
			for (int index=N; index<MinibatchSize; ++index)
			{
				std::fill(target, target + InputDataSize, 0);
				target += InputDataSize;
			}
	  
			net.feeder.input(frames_input);
			net.net->ForwardPrefilled(nullptr);

			int index = 0;
			std::generate(policies.begin(), policies.end(), [&](){ return get_policy(index++,is_valid_action); });
			return policies;
		}

		Policy get_policy(int index,std::function<bool(int)> is_valid_action)
		{
			auto q_at = [&](int action){
				auto q =  net.q_values_blob->data_at(index,action,0,0);
				assert(is_valid_q(q));
				return q;
			};

			Policy best(nullptr);
			
			for (int action=0; action<num_actions; ++action)
			{
				if (is_valid_action(action))
				{
					auto q = q_at(action);
					if (q > best.val)
					{
						best = Policy(action,q);
					}				
				}				
			}	
			
			// std::cout << str(format("%d:%.2f")%best.action%best.val) << "\n";
			return best;
		}
	};

	class Trainer
	{
	public :
		DeepNetwork& net;
		Trainer(DeepNetwork& net) : net(net) {}

		FramesLayerInputData frames_input;
  		TargetLayerInputData target_input;
  		FilterLayerInputData filter_input;

  		std::array<const Experience*,MinibatchSize> samples;
		std::array<InputFrames,MinibatchSize> input_frames_batch;		

		void train()
		{			
			for (int k=0; k<MinibatchSize; ++k)
			{
				const auto& e = net.replay_memory.get_random();
				e.check_sanity();

				samples[k] = &e;

				if (e.next_frame) 
				{
					for (int j=0; j<temporal_window; ++j)
					{
						input_frames_batch[k][j] = e.input_frames[j+1];						
					}				
					input_frames_batch[k][temporal_window] = e.next_frame;
				}	
			}

			const auto& policies = net.eval_for_train.evaluate(input_frames_batch,[=](int action){return true;});

			auto frame = frames_input.begin();
			auto target = target_input.begin();
			auto filter = filter_input.begin();

			std::fill(target_input.begin(),target_input.end(),0);
			std::fill(filter_input.begin(),filter_input.end(),0);
  	
			for (int index=0; index<MinibatchSize; ++index)
			{
				const auto e = samples[index];
				const auto& p = policies[index];

				float r = e->next_frame ? e->reward + net.gamma * p.val : e->reward;			

				e->check_sanity();
				assert(is_valid_q(r));

				SingleFrame::fill_frames(frame,e->input_frames);
				
				target[e->action] = r;
				filter[e->action] = 1.0f;
				
				target += num_actions;
				filter += num_actions;
			}

			net.feeder.input(frames_input,target_input,filter_input);

			net.solver->Step(1);
		}
	};

	void check_sanity()
	{
		for (auto layer : net->layers())
		{
			auto& blobs = layer->blobs();
			if (blobs.size())
			{
				auto blob = blobs.front();
				assert(blob);
				auto v = blob->data_at(0, 0, 0, 0);
				assert(!std::isnan(v));
			}
		}
	}

	typedef shared_ptr<caffe::Blob<float>> BlobSp;
	typedef shared_ptr<caffe::Net<float>> NetSp;
	typedef shared_ptr<caffe::Solver<float>> SolverSp;

	Feeder feeder;
	ReplayMemory replay_memory;	
	
	Evaluator<1> eval_for_prediction;
	Evaluator<MinibatchSize> eval_for_train;
	Trainer trainer;

	BlobSp q_values_blob;
	NetSp net;
	SolverSp solver;
	mutable std::mt19937 random_engine;

	AnnealedEpsilon epsilon;	
	float gamma;	

	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
	
	DeepNetwork()
	: gamma(FLAGS_gamma), trainer(*this), eval_for_prediction(*this), eval_for_train(*this), replay_memory(*this), feeder(*this)
	{		
		net_init();
	}

	void load_trained(const std::string& model_bin)
	{
		net->CopyTrainedLayersFrom(model_bin);
	}

	void net_init()
	{
		net_create();

		feeder.init();
	}

	void net_create()
	{
		caffe::SolverParameter solver_param;
		LOG(INFO) << FLAGS_solver;
		caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
		
		solver.reset(caffe::GetSolver<float>(solver_param));
		net = solver->net();
		q_values_blob = net->blob_by_name("q_values");
	}

	Policy policy(const InputFrames& input_frames,std::function<bool(int)> is_valid_action)
	{
		return eval_for_prediction.evaluate(std::array<InputFrames,1>{{input_frames}},is_valid_action).front();
	}	

	Policy predict(const InputFrames& input_frames,std::function<int()> random_action,std::function<bool(int)> is_valid_action)
	{		
		if (epsilon.should_do_random_action())
		{
			return Policy(random_action());
		}
		else
		{
			Policy p = policy(input_frames,is_valid_action);
			if (is_valid_action(p.action))
			{
				return p;
			}
			else
			{
				std::cout << "couldn't predict well\n";
				return Policy(random_action());
			}
		}
	}		

	void train()
	{
		if (replay_memory.has_enough())
		{
			trainer.train();
		}
	}	
};

#include "brain.h"