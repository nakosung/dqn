// fix
enum { batch_size = 32 };
enum { temporal_window = 3 };	
enum { world_size = 16 };

// var per
enum { max_actions = 5 };
enum { num_states = world_size * world_size };
enum { num_actions = max_actions };
enum { net_inputs = (num_states + num_actions) * temporal_window + num_states };

enum { MinibatchSize = batch_size };
enum { InputDataSize = num_states };
enum { MinibatchDataSize = InputDataSize * MinibatchSize };
enum { OutputCount = num_actions };

typedef std::vector<float> net_input_type;
typedef std::array<float,num_states> input_type;
typedef std::array<input_type,temporal_window+1> State;

struct Frame
{
	input_type input;
	State state;
	int action;
	float reward;
};

struct Experience : Frame
{
	State next_state;
};

struct Policy
{
	int action;
	float val;
};

class AnnealedEpsilon
{
public:
	float epsilon, epsilon_min, epsilon_test_time;
	int age;
	bool is_learning;
	int learning_steps_total, learning_steps_burnin;

	AnnealedEpsilon()
	: is_learning(true), age(0)
	{}

	float get() const
	{
		if (is_learning)
		{
			return std::min(1.0f,std::max(epsilon_min, 1.0f-(age - learning_steps_burnin)/(learning_steps_total - learning_steps_burnin)));
		}
		else
		{
			return epsilon_test_time;
		}
	}

	void operator ++()
	{
		age++;
	}
};

class DeepNetwork
{
public :
	typedef std::array<float,MinibatchDataSize> FramesLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> TargetLayerInputData;
	typedef std::array<float,MinibatchSize * OutputCount> FilterLayerInputData;

	typedef shared_ptr<caffe::Blob<float>> BlobSp;
	typedef shared_ptr<caffe::Net<float>> NetSp;
	typedef shared_ptr<caffe::Solver<float>> SolverSp;
	typedef shared_ptr<caffe::MemoryDataLayer<float>> MemoryDataLayerSp;

	BlobSp q_values_blob;
	NetSp net;
	SolverSp solver;
	MemoryDataLayerSp frames_input_layer;
	MemoryDataLayerSp target_input_layer;
	MemoryDataLayerSp filter_input_layer;
	TargetLayerInputData dummy_input_data;
	mutable std::mt19937 random_engine;

	AnnealedEpsilon epsilon;
	std::vector<Experience> experiences;
	int experience_size;
	int start_learn_threshold;
	float gamma;

	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
	
	DeepNetwork()
	: experience_size(30000)
	{
		start_learn_threshold = std::max(experience_size / 10, 1000);
		experiences.reserve(experience_size);

		net_init();
	}

	void net_init()
	{
		caffe::SolverParameter solver_param;
		LOG(INFO) << FLAGS_solver;
		caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
		
		solver.reset(new caffe::SGDSolver<float>(solver_param));
		net = solver->net();
		q_values_blob = net->blob_by_name("q_values");
		std::fill(dummy_input_data.begin(),dummy_input_data.end(),0.0);
	}

	std::vector<Policy> net_batch_policy(const std::vector<State>& states)
	{
		std::array<float,MinibatchDataSize> frames;
		// fill
		net_input(frames,dummy_input_data,dummy_input_data);
		net->ForwardPrefilled(nullptr);

		return std::vector<Policy>{{0,1}};
	}

	Policy policy(State s)
	{
		std::array<float,MinibatchDataSize> frames_input;
		
		return net_batch_policy(std::vector<State>{{s}}).front();
	}

	bool test_epsilon()
	{
		float dice = std::uniform_real_distribution<float>(0,1)(random_engine);
		return dice < epsilon.get();
	}

	int predict(State s,std::function<int()> random_action)
	{
		if (test_epsilon())
		{
			return random_action();
		}
		else
		{
			return policy(s).action;
		}
	}

	Experience& push_experience()
	{
		++epsilon;

		Experience* out;

		// being slow for limited time frames
		if (experiences.size() < experience_size)
		{
			Experience e;
			experiences.push_back(e);
			out = &experiences.back();
		}
		else
		{
			out = &experiences[randint(experience_size)];
		}

		return *out;
	}

	void train()
	{
		if (experiences.size() > start_learn_threshold)
		{
			FramesLayerInputData frames_input;
  			TargetLayerInputData target_input;
  			FilterLayerInputData filter_input;

			std::vector<const Experience*> samples;
			std::vector<State> states;
			for (int k=0; k<batch_size; ++k)
			{
				int re = randint(experience_size);
				samples.push_back(&experiences[re]);
				states.push_back(experiences[re].state);
			}

			const auto policies = net_batch_policy(states);
			int index = 0;

			auto target = frames_input.begin();
  	
			for (auto p: policies)
			{
				const Experience& e = *samples[index];
				float r = e.reward + gamma * p.val;

				for (const auto& frame : e.state)
				{	
					std::copy(frame.begin(),frame.end(),target);
					target += frame.size();
				}
				target_input[index] = r;
				filter_input[index] = 1.0f;

				index++;
			}

			net_input(frames_input,target_input,filter_input);

			solver->Step(1);
		}
	}

	void net_input( const FramesLayerInputData& frames, const TargetLayerInputData& target, const FilterLayerInputData& filter )
	{
		frames_input_layer->Reset(const_cast<float*>(frames.data()),dummy_input_data.data(),MinibatchSize);
		target_input_layer->Reset(const_cast<float*>(target.data()),dummy_input_data.data(),MinibatchSize);
		filter_input_layer->Reset(const_cast<float*>(filter.data()),dummy_input_data.data(),MinibatchSize);
	}
};

class Brain
{
public:
	typedef boost::shared_ptr<DeepNetwork> NetworkSp;

	int forward_passes;
	float latest_reward;
	
	NetworkSp network;

	Brain(NetworkSp network)
	: forward_passes(0), latest_reward(0), network(network)
	{}

	int forward(const input_type& input_array,std::function<int()> random_action)
	{
		forward_passes++;
		
		Frame frame;

		frame.input = input_array;

		if (forward_passes > temporal_window)
		{
			frame.state = get_net_input(input_array);
			frame.action = network->predict(frame.state,random_action);
		}
		else
		{
			frame.action = random_action();
		}

		frame_window.pop_front();
		frame_window.push_back(frame);

		return frame.action;
	}

	void backward(float reward)
	{
		latest_reward = reward;
		auto itr = frame_window.rbegin();
		Frame& cur = *itr++;
		cur.reward = reward;

		if (!network->epsilon.is_learning) return;

		if (forward_passes > temporal_window + 1)
		{
			const Frame& prev = *itr;

			Experience& e = network->push_experience();
			e.state = prev.state;
			e.action = prev.action;
			e.reward = prev.reward;
			e.next_state = cur.state;
			
			network->train();
		}
	}

	static net_input_type action1ofk(int N, int k)
	{
		net_input_type v;
		v.resize(N);
		v[k] = 1.0f;
		return v;
	}


	State get_net_input(const input_type& xt)
	{
		State w;
		w[0] = xt;
		
		for (int k=0; k<temporal_window; ++k)
		{
			w[k+1] = frame_window[k].input;
			// append(w,action1ofk(num_actions,frame_window[k].action));
		}

		return w;
	}	

	std::deque<Frame> frame_window;
};