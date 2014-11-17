// fix
enum { batch_size = 32 };
enum { temporal_window = 3 };	
enum { world_size = 16 };

// var per
enum { num_states = world_size * world_size };
enum { num_actions = 5 };
enum { net_inputs = (num_states + num_actions) * temporal_window + num_states };

enum { MinibatchSize = batch_size };
enum { InputDataSize = num_states * (temporal_window + 1) };
enum { MinibatchDataSize = InputDataSize * MinibatchSize };
enum { OutputCount = num_actions };

typedef std::array<float,num_actions> net_input_type;
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

	AnnealedEpsilon()
	: is_learning(true), age(0), epsilon_min(0.1), learning_steps_burnin(3000), learning_steps_total(100000), epsilon_test_time(0.1)
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
	std::vector<Experience> experiences; // reserved, no fragmentation
	int experience_size;
	int start_learn_threshold;
	float gamma;

	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
	
	DeepNetwork()
	: experience_size(30000), gamma(0.95), trainer(*this), eval_for_prediction(*this), eval_for_train(*this)
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

		auto get_layer = [&](const char* name){
			auto result = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name(name));
			assert(result);
			return result;
		};

		auto check_blob_size = [&](const char* blob_name,int n,int c,int h,int w) {
			auto blob = net->blob_by_name(blob_name);
			assert(blob);
			assert(blob->num() == n);
			assert(blob->channels() == c);
			assert(blob->height() == h);
			assert(blob->width() == w);
		};

		frames_input_layer = get_layer("frames_input_layer");		
		target_input_layer = get_layer("target_input_layer");
		filter_input_layer = get_layer("filter_input_layer");

		check_blob_size("frames",MinibatchSize,temporal_window+1,world_size,world_size);
		check_blob_size("target",MinibatchSize,num_actions,1,1);
		check_blob_size("filter",MinibatchSize,num_actions,1,1);
	}

	template <int N>
	class Evaluator
	{
	public:
		DeepNetwork& net;
		Evaluator(DeepNetwork& net) : net(net) {}

		std::array<Policy,N> policies;
		std::array<Policy,N>& evaluate(const std::array<State,N>& states)
		{
			assert(states.size() <= MinibatchSize);
			
			FramesLayerInputData frames_input;
			auto target = frames_input.begin();
			for (const auto& state : states)
			{
				for (const auto& input : state)
				{
					std::copy(input.begin(),input.end(),target);
					target += input.size();
				}
			}  
	  
			net.net_input(frames_input,net.dummy_input_data,net.dummy_input_data);
			net.net->ForwardPrefilled(nullptr);			

			int index = 0;
			for (const auto& state : states)
			{
				std::array<float,num_actions> q_values;
				int action = 0;
				std::generate(q_values.begin(),q_values.end(),[&]{
					return net.q_values_blob->data_at(index, action, 0, 0);
				});

				auto max_elem = std::max_element(q_values.begin(),q_values.end());
				const auto max_idx = std::distance(q_values.begin(),max_elem);
				policies[index].action = action;
				policies[index].val = *max_elem;

				index++;
			}

			return policies;
		}
	};


	// std::vector<Policy> net_batch_policy(const std::vector<State>& states)
	// {
	// 	assert(states.size() <= MinibatchSize);
		
	// 	FramesLayerInputData frames_input;
	// 	auto target = frames_input.begin();
	// 	for (const auto& state : states)
	// 	{
	// 		for (const auto& input : state)
	// 		{
	// 			std::copy(input.begin(),input.end(),target);
	// 			target += input.size();
	// 		}
	// 	}  
  
	// 	net_input(frames_input,dummy_input_data,dummy_input_data);
	// 	net->ForwardPrefilled(nullptr);

	// 	std::vector<Policy> policies;
	// 	policies.reserve(states.size());

	// 	int index = 0;
	// 	for (const auto& state : states)
	// 	{
	// 		std::array<float,num_actions> q_values;
	// 		int action = 0;
	// 		std::generate(q_values.begin(),q_values.end(),[&]{
	// 			return q_values_blob->data_at(index, action, 0, 0);
	// 		});

	// 		auto max_elem = std::max_element(q_values.begin(),q_values.end());
	// 		const auto max_idx = std::distance(q_values.begin(),max_elem);
	// 		policies.emplace_back(action,*max_elem);

	// 		index++;
	// 	}			

	// 	return policies;
	// }

	Policy policy(State s)
	{
		std::array<float,MinibatchDataSize> frames_input;
		
		return eval_for_prediction.evaluate(std::array<State,1>{{s}}).front();
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

	class Trainer
	{
	public :
		DeepNetwork& net;
		Trainer(DeepNetwork& net) : net(net) {}

		FramesLayerInputData frames_input;
  		TargetLayerInputData target_input;
  		FilterLayerInputData filter_input;

  		std::array<const Experience*,batch_size> samples;
		std::array<State,batch_size> states;		

		void train()
		{
			for (int k=0; k<batch_size; ++k)
			{
				int re = net.randint(net.experience_size);
				samples[k] = &net.experiences[re];
				states[k] = net.experiences[re].state;
			}

			const auto& policies = net.eval_for_train.evaluate(states);

			auto frame = frames_input.begin();
			auto target = target_input.begin();
			auto filter = filter_input.begin();
  	
  			int index=0;
			for (auto p: policies)
			{
				const auto e = samples[index];
				float r = e->reward + net.gamma * p.val;

				for (const auto& f : e->state)
				{	
					std::copy(f.begin(),f.end(),frame);
					frame += f.size();
				}
				*target++ = r;
				*filter++ = 1.0f;
				index++;
			}

			net.net_input(frames_input,target_input,filter_input);

			net.solver->Step(1);
		}
	};

	Evaluator<1> eval_for_prediction;
	Evaluator<MinibatchSize> eval_for_train;

	Trainer trainer;

	void train()
	{
		// LOG(INFO) << "training";

		if (experiences.size() > start_learn_threshold)
		{
			trainer.train();
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
	
	NetworkSp network;

	Brain()
	: forward_passes(0)
	{}

	int forward(const input_type& input_array,std::function<int()> random_action)
	{
		forward_passes++;

		// LOG(INFO) << "forwarding pass:" << forward_passes;
		
		Frame frame;

		frame.input = input_array;

		if (forward_passes > temporal_window)
		{
			// LOG(INFO) << "getting network input";
			frame.state = get_net_input(input_array);

			// LOG(INFO) << "predicting action";
			frame.action = network->predict(frame.state,random_action);

			// LOG(INFO) << "predict completed";			
			frame_window.pop_front();
		}
		else
		{
			// LOG(INFO) << "random action";
			frame.action = random_action();
		}
		
		frame_window.push_back(frame);

		// LOG(INFO) << "returning from forward";

		return frame.action;
	}

	void backward(float reward)
	{
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
		std::fill(v.begin(),v.end(),0.0);
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