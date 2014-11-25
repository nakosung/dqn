#include "environment.h"
#include "config.h"
#include "single_frame.h"
#include "google/protobuf/text_format.h"
#include "caffe/proto/caffe.pb.h"
#include <fstream>
#include <streambuf>
#include <unordered_map>

DEFINE_int32(experience_size, 10, "experience_size percent");
DEFINE_int32(learning_steps_total, 500000, "learning_steps_total");
DEFINE_int32(learning_steps_burnin, -1, "learning_steps_burnin");
DEFINE_int32(epsilon_min, 0.1, "epsilon_min");
DEFINE_int32(epsilon_test, 0.05, "epsilon_test");
DEFINE_double(gamma, 0.95, "gamma");
DEFINE_int32(display_interval, 1, "display_interval");
DEFINE_int32(display_after, 10000, "display_after");

typedef std::array<float,num_actions> net_input_type;
typedef boost::shared_ptr<SingleFrame> SingleFrameSp;
typedef std::array<SingleFrameSp,window_length> InputFrames;

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
	Environment& env;
	float epsilon, epsilon_min, epsilon_test_time;
	int age;
	bool is_learning;
	int learning_steps_total, learning_steps_burnin;	

	AnnealedEpsilon(Environment& env)
	: env(env), is_learning(true), age(0), epsilon_min(FLAGS_epsilon_min), learning_steps_burnin(FLAGS_learning_steps_burnin < 0 ? FLAGS_learning_steps_total / 10 : FLAGS_learning_steps_burnin), learning_steps_total(FLAGS_learning_steps_total), epsilon_test_time(FLAGS_epsilon_test)
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
		const float eps = get();
		assert(is_valid_epsilon(eps));
		return env.test_prob(eps);
	}

	void operator ++()
	{		
		age++;
	}
};

class DeepNetwork
{
public :
	Environment& env;

	typedef std::array<float,MinibatchSize * InputDataSize> FramesLayerInputData;
	typedef std::array<float,MinibatchSize * StatChannels> StatsLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> TargetLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> FilterLayerInputData;

	typedef std::function<bool(int)> IsValidActionFunctionType;
	typedef std::function<int()> RandomActionFunctionType;

	typedef shared_ptr<caffe::Blob<float>> BlobSp;
	typedef shared_ptr<caffe::Net<float>> NetSp;
	typedef shared_ptr<caffe::Solver<float>> SolverSp;

	class ReplayMemory
	{
	public :
		DeepNetwork& net;

		ReplayMemory(DeepNetwork& net)
		: size(std::max(0,std::min(100,FLAGS_experience_size)) * FLAGS_learning_steps_total / 100), net(net)
		{
			experiences.reserve(size);
		}

		bool has_more_than( size_t num_experiences ) const
		{
			return experiences.size() > num_experiences;
		}

		const Experience& get_random() const
		{
			return experiences[ net.env.randint(experiences.size()) ];
		}

		void push(const Experience& e)
		{
			Experience* out;

			// being slow for limited time frames
			if (experiences.size() < size)
			{
				experiences.push_back(e);				
			}
			else
			{
				experiences[net.env.randint(size)] = e;
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
		MemoryDataLayerSp stats_input_layer;
		MemoryDataLayerSp target_input_layer;
		MemoryDataLayerSp filter_input_layer;
		TargetLayerInputData dummy_input_data;

		class Cursor
		{
		public:
			FramesLayerInputData frames_input;
			StatsLayerInputData stats_input;
	  		TargetLayerInputData target_input;	  		
	  		FilterLayerInputData filter_input;
	  		Feeder& feeder;

			Cursor(Feeder& feeder) : feeder(feeder)
			{}

			FramesLayerInputData::iterator frames;
			StatsLayerInputData::iterator stats;
			TargetLayerInputData::iterator target;
			FilterLayerInputData::iterator filter;

			void begin()
			{
				frames = frames_input.begin();
				stats = stats_input.begin();
				target = target_input.begin();
				filter = filter_input.begin();

				std::fill(target_input.begin(),target_input.end(),0);
				std::fill(filter_input.begin(),filter_input.end(),0);
			}

			template <typename U>
			void write_frames(const U& input_frames)
			{				
				auto target = frames;
				auto target_stats = stats;
				for (const auto& f : input_frames)
				{	
					auto frame = f.get();
					if (frame)
					{
						for (const auto& image : frame->images)
						{
							std::copy(image.begin(),image.end(),target);
							assert(image.size() == sight_area);
							target += sight_area;
						}						

						std::copy(frame->stats.begin(), frame->stats.end(), target_stats);
					}
					else
					{
						std::fill(target, target + ImageSize,0);
						target += ImageSize;
					}										
					target_stats += num_stats;
				}
			}

			void write_target(int action, float r)
			{
				target[action] = r;
				filter[action] = 1.0f;
			}

			void advance()
			{
				frames += InputDataSize;
				stats += StatChannels;
				target += OutputCount;
				filter += OutputCount;
			}

			void done()
			{
				std::fill(frames, frames_input.end(), 0);
				std::fill(stats, stats_input.end(), 0);
				
				feeder.input(frames_input,stats_input,target_input,filter_input);
			}
		};		

		Feeder(DeepNetwork& net)
		: net(net)
		{		
			init();
		}

		void init()
		{
			cache_layers();
			check_sanity();

			std::fill(dummy_input_data.begin(),dummy_input_data.end(),0.0);
		}

		void input( const FramesLayerInputData& frames, const StatsLayerInputData& stats, const TargetLayerInputData& target, const FilterLayerInputData& filter )
		{
			frames_input_layer->Reset(const_cast<float*>(frames.data()),dummy_input_data.data(),MinibatchSize);
			stats_input_layer->Reset(const_cast<float*>(stats.data()),dummy_input_data.data(),MinibatchSize);
			target_input_layer->Reset(const_cast<float*>(target.data()),dummy_input_data.data(),MinibatchSize);
			filter_input_layer->Reset(const_cast<float*>(filter.data()),dummy_input_data.data(),MinibatchSize);
		}		

		void forward()
		{
			net.net->ForwardPrefilled(nullptr);			
		}

		void cache_layers()
		{
			auto get_layer = [&](const char* name){
				auto result = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net.net->layer_by_name(name));
				assert(result);
				return result;
			};		

			frames_input_layer = get_layer("frames_input_layer");		
			stats_input_layer = get_layer("stats_input_layer");		
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

			check_blob_size("frames",MinibatchSize,ImageChannels,sight_diameter,sight_diameter);
			check_blob_size("stats",MinibatchSize,StatChannels,1,1);
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
		BlobSp q_values_blob;
		Feeder::Cursor cursor;

		Evaluator(DeepNetwork& net) : net(net), cursor(net.feeder)
		{
			init();
		}

		void init()
		{
			q_values_blob = net.net->blob_by_name("q_values");			
		}

		std::array<Policy,N> policies;
		std::array<Policy,N>& evaluate(const std::array<InputFrames,N>& input_frames_batch,IsValidActionFunctionType is_valid_action)
		{			
			cursor.begin();
			for (const auto& input_frames : input_frames_batch)
			{
				cursor.write_frames(input_frames);
				cursor.advance();
			}  			

			cursor.done();
	  
			net.feeder.forward();

			int index = 0;
			std::generate(policies.begin(), policies.end(), [&](){ return get_policy(index++,is_valid_action); });
			return policies;
		}

		Policy get_policy(int index, IsValidActionFunctionType is_valid_action)
		{
			auto q_at = [&](int action){
				auto q = q_values_blob->data_at(index,action,0,0);
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
		float gamma;		

		Feeder::Cursor cursor;

  		ReplayMemory replay_memory;	

  		std::array<const Experience*,MinibatchSize> samples;
		std::array<InputFrames,MinibatchSize> input_frames_batch;

		BlobSp loss_blob;

		Trainer(DeepNetwork& net) : net(net), gamma(FLAGS_gamma), replay_memory(net), cursor(net.feeder)
		{
			init();
		}

		void init()
		{
			loss_blob = net.net->blob_by_name("loss");
		}

		void push(const Experience& e)
		{
			++net.epsilon;
			
			replay_memory.push(e);
		}

		void train()
		{			
			if (!replay_memory.has_more_than(net.epsilon.learning_steps_burnin))
			{
				return;
			}		
		
			for (int k=0; k<MinibatchSize; ++k)
			{
				const auto& e = replay_memory.get_random();
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

			cursor.begin();
			
			for (int index=0; index<MinibatchSize; ++index)
			{
				const auto e = samples[index];
				const auto& p = policies[index];

				float r = e->next_frame ? e->reward + gamma * p.val : e->reward;			

				e->check_sanity();
				assert(is_valid_q(r));

				cursor.write_frames(e->input_frames);
				cursor.write_target(e->action,r);
				
				cursor.advance();
			}

			cursor.done();

			net.solver->Step(1);			
		}
	};

	class Loader
	{
	public :
		DeepNetwork& net;

		Loader(DeepNetwork& net, std::string file) : net(net)
		{
			init(file);						
		}

		std::string read_text(std::string file)
		{
			std::ifstream t(file);
			return std::string((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
		}
		
		void init(std::string file)
		{
			caffe::SolverParameter param;
			LOG(INFO) << FLAGS_solver;
			
			std::string proto = read_text(file);			
			replace_proto(proto);

			CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));

			switch (Caffe::mode()) {
			case Caffe::CPU:
				param.set_solver_mode(caffe::SolverParameter_SolverMode_CPU);
				break;
			case Caffe::GPU:
				param.set_solver_mode(caffe::SolverParameter_SolverMode_GPU);
				break;
			default:
				LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
			}
			
			net.solver.reset(caffe::GetSolver<float>(param));
			net.net = net.solver->net();
		}

		void load_trained(const std::string& model_bin)
		{
			net.net->CopyTrainedLayersFrom(model_bin);
			net.epsilon.is_learning = false;
			net.solver.reset();
		}

	private:
		void replace_proto(std::string& proto)
		{
			std::unordered_map<std::string, std::string> dictionary;

			auto dict_add_int = [&](std::string key, int value) {
				dictionary[key] = str(format("%d")%value);
			};

			dict_add_int("BATCH_SIZE",MinibatchSize);
			dict_add_int("HIDDEN_LAYER_SIZE",HiddenLayerSize);
			dict_add_int("IMAGE_FEATURE_SIZE",ImageFeatureSize);
			dict_add_int("LOWLEVEL_IMAGE_FEATURE_SIZE",LowLevelImageFeatureSize);
			dict_add_int("NUM_ACTIONS",num_actions);
			dict_add_int("SIGHT_SIZE",sight_diameter);
			dict_add_int("IMAGE_CHANNELS",ImageChannels);
			dict_add_int("STAT_CHANNELS",StatChannels);
			dict_add_int("LOWLEVEL_KERNEL_SIZE",LowLevelKernelSize);
			dict_add_int("LOWLEVEL_KERNEL_STRIDE",LowLevelKernelSize / 2);
			dict_add_int("KERNEL_SIZE",KernelSize);
			dict_add_int("KERNEL_STRIDE",KernelSize / 2);

			for (;;)
			{
				auto found = proto.find("{{");
				if (found == std::string::npos) break;

				auto closing = proto.find("}}");
				if (closing == std::string::npos) break;

				if (found > closing) break;

				auto key = proto.substr(found+2,closing-found-2);
				proto = proto.replace(found,closing-found+2,dictionary[key]);
			}
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

	AnnealedEpsilon epsilon;		
	NetSp net;
	SolverSp solver;	

	Loader loader;
	Feeder feeder;		
	Evaluator<1> eval_for_prediction;
	Evaluator<MinibatchSize> eval_for_train;
	Trainer trainer;
	
	DeepNetwork(Environment& env,std::string file)
	: env(env), loader(*this,file), epsilon(env), trainer(*this), eval_for_prediction(*this), eval_for_train(*this), feeder(*this)
	{}		

	Policy predict(const InputFrames& input_frames,RandomActionFunctionType random_action,IsValidActionFunctionType is_valid_action)
	{	
		if (epsilon.should_do_random_action())
		{
			return Policy(random_action());
		}
		else
		{
			return eval_for_prediction.evaluate(std::array<InputFrames,1>{{input_frames}},is_valid_action).front();
		}
	}		

	void train()
	{
		trainer.train();		
	}	
};

#include "brain.h"