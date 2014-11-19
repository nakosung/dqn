// fix
enum { temporal_window = 3 };	
enum { world_size = 16 };

// var per
enum { num_states = world_size * world_size };
enum { num_actions = 6 };

enum { MinibatchSize = 32 };
enum { InputDataSize = num_states * (temporal_window + 1) };
enum { OutputCount = num_actions };

typedef std::array<float,num_actions> net_input_type;
typedef std::array<float,num_states> SingleFrame;
typedef boost::shared_ptr<SingleFrame> SingleFrameSp;
typedef std::array<SingleFrameSp,temporal_window+1> InputFrames;

DEFINE_int32(experience_size, 30000, "experience_size");
DEFINE_int32(learning_steps_burnin, -1, "learning_steps_burnin");
DEFINE_int32(learning_steps_total, 200000, "learning_steps_total");
DEFINE_int32(epsilon_min, 0.1, "epsilon_min");

struct Experience
{
	InputFrames input_frames;
	int action;
	float reward;
	SingleFrameSp next_frame;	

	void check_sanity()
	{
		assert(reward < 100 && reward > -100 );
		assert(action >= 0 && action < num_actions);
	}
};

struct Policy
{
	int action;
	float val;

	Policy() {}
	Policy(int action,float val) : action(action), val(val) {}
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
		float dice = std::uniform_real_distribution<float>(0,1)(random_engine);		
		return dice < get();
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
			assert(std::all_of(frames.begin(),frames.end(),[=](float x){return !std::isnan(x);}));
			assert(std::all_of(target.begin(),target.end(),[=](float x){return !std::isnan(x);}));
			assert(std::all_of(filter.begin(),filter.end(),[=](float x){return !std::isnan(x);}));
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

			check_blob_size("frames",MinibatchSize,temporal_window+1,world_size,world_size);
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
		std::array<Policy,N>& evaluate(const std::array<InputFrames,N>& input_frames_batch)
		{			
			FramesLayerInputData frames_input;
			auto target = frames_input.begin();			
			for (const auto& input_frames : input_frames_batch)
			{
				for (const auto& input_frame : input_frames)
				{				
					if (input_frame)
					{
						std::copy(input_frame->begin(),input_frame->end(),target);						
						assert(input_frame->size() == num_states);
					}	
					else
					{
						std::fill(target, target + num_states, 0);
					}					
					target += num_states;
				}
				assert(input_frames.size() == temporal_window+1);
			}  
			for (int index=N; index<MinibatchSize; ++index)
			{
				std::fill(target, target + InputDataSize, 0);
				target += InputDataSize;
			}
			assert(target == frames_input.end());
	  
			net.feeder.input(frames_input);
			net.net->ForwardPrefilled(nullptr);

			int index = 0;
			std::generate(policies.begin(), policies.end(), [&](){ return get_policy(index++); });
			return policies;
		}

		Policy get_policy(int index)
		{
			auto q_at = [&](int action){return net.q_values_blob->data_at(index,action,0,0);};

			Policy best(0,q_at(0));

			if (net.needs_explanation)
			{
				net.explanation = str(format("%.2f")%best.val);
			}
			
			for (int action=1; action<num_actions; ++action)
			{
				auto q = q_at(action);
				if (q > best.val)
				{
					best = Policy(action,q);
				}
				if (net.needs_explanation)
				{
					net.explanation += str(format(",%.2f")%q);
				}
			}			
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
				auto e = net.replay_memory.get_random();
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

			const auto& policies = net.eval_for_train.evaluate(input_frames_batch);

			auto frame = frames_input.begin();
			auto target = target_input.begin();
			auto filter = filter_input.begin();

			std::fill(frames_input.begin(),frames_input.end(),0);
			std::fill(target_input.begin(),target_input.end(),0);
			std::fill(filter_input.begin(),filter_input.end(),0);
  	
			for (int index=0; index<MinibatchSize; ++index)
			{
				const auto e = samples[index];
				const auto p = policies[index];

				float r = e->next_frame ? e->reward + net.gamma * p.val : e->reward;			

				assert(!std::isnan(e->reward));
				assert(!e->next_frame || !std::isnan(p.val));
				assert(!std::isnan(r));
				
				for (const auto& f : e->input_frames)
				{	
					std::copy(f->begin(),f->end(),frame);
					frame += f->size();
				}
				target[e->action] = r;
				filter[e->action] = 1.0f;

				target += num_actions;
				filter += num_actions;
			}

			assert(target == target_input.end());
			assert(frame == frames_input.end());
			assert(filter == filter_input.end());

			net.feeder.input(frames_input,target_input,filter_input);

			// net.check_sanity();

			// LOG(INFO) << &net;

			// net.check_sanity();

			net.solver->Step(1);

			// net.check_sanity();

		// 	auto net_ = net.net;

		// 	LOG(INFO) << "conv1:" <<
  //     net_->layer_by_name("conv1_layer")->blobs().front()->data_at(1, 0, 0, 0);
  // LOG(INFO) << "conv2:" <<
  //     net_->layer_by_name("conv2_layer")->blobs().front()->data_at(1, 0, 0, 0);
  // LOG(INFO) << "ip1:" <<
  //     net_->layer_by_name("ip1_layer")->blobs().front()->data_at(1, 0, 0, 0);
  // LOG(INFO) << "ip2:" <<
  //     net_->layer_by_name("ip2_layer")->blobs().front()->data_at(1, 0, 0, 0);
		}
	};

	void check_sanity()
	{
		// LOG(INFO) << "checking!";
		auto check_layer = [&](const char* name) {
			// std::cout << name << v;
			auto layer = net->layer_by_name(name);
			assert(layer);
			auto& blobs = layer->blobs();
			assert(blobs.size());
			auto blob = blobs.front();
			assert(blob);
			auto v = blob->data_at(1, 0, 0, 0);
			assert(!std::isnan(v));
		};

		check_layer("conv1_layer");
		check_layer("conv2_layer");
		check_layer("ip1_layer");
		check_layer("ip2_layer");
	}

	typedef shared_ptr<caffe::Blob<float>> BlobSp;
	typedef shared_ptr<caffe::Net<float>> NetSp;
	typedef shared_ptr<caffe::Solver<float>> SolverSp;

	Feeder feeder;
	ReplayMemory replay_memory;
	bool needs_explanation;
	std::string explanation;

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
	: gamma(0.95), trainer(*this), eval_for_prediction(*this), eval_for_train(*this), replay_memory(*this), feeder(*this), needs_explanation(false)
	{		
		net_init();
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
		
		solver.reset(new caffe::SGDSolver<float>(solver_param));
		net = solver->net();
		q_values_blob = net->blob_by_name("q_values");
	}

	Policy policy(const InputFrames& input_frames)
	{
		return eval_for_prediction.evaluate(std::array<InputFrames,1>{{input_frames}}).front();
	}	

	int predict(const InputFrames& input_frames,std::function<int()> random_action)
	{		
		if (epsilon.should_do_random_action())
		{
			return random_action();
		}
		else
		{
			return policy(input_frames).action;
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

class Brain
{
public:
	typedef boost::shared_ptr<DeepNetwork> NetworkSp;

	int forward_passes;
	
	NetworkSp network;
	Experience current_experience;

	bool has_pending_experience;

	Brain()
	: forward_passes(0), has_pending_experience(false)
	{}

	void flush(SingleFrameSp next_frame)
	{
		if (has_pending_experience)
		{
			if (!(current_experience.reward < 100))
			{
				std::cout << "invalid reward: " << current_experience.reward;
				exit(-1);
			}
			current_experience.next_frame = next_frame;					
			network->replay_memory.push(current_experience);				
			network->train();

			has_pending_experience = false;
		}
	}

	std::string q_values_str;
	std::string detail() const { return q_values_str; }

	virtual bool needs_explanation() const { return false; }

	int forward(SingleFrameSp frame,std::function<int()> random_action)
	{
		forward_passes++;
		
		flush(frame);

		if (forward_passes > temporal_window + 1 && network->epsilon.is_learning)
		{
			has_pending_experience = true;
			std::copy(frame_window.begin(), frame_window.end(), current_experience.input_frames.begin());
			network->needs_explanation = needs_explanation();
			current_experience.action = network->predict(current_experience.input_frames,random_action);
			network->needs_explanation = false;
			q_values_str = network->explanation;

			frame_window.pop_front();
		}
		else
		{
			// LOG(INFO) << "random action";
			current_experience.action = random_action();
		}
		
		frame_window.push_back(frame);
		
		return current_experience.action;
	}

	void backward(float reward)
	{
		current_experience.reward = reward;		
	}	
	
	std::deque<SingleFrameSp> frame_window;
};