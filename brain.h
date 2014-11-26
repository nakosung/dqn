class Brain
{
public:
	typedef boost::shared_ptr<DeepNetwork> NetworkSp;

	int forward_passes;
	
	NetworkSp network;
	Experience current_experience;

	bool has_pending_experience;

	Brain(NetworkSp network)
	: forward_passes(0), has_pending_experience(false), network(network)
	{
		last_non_random_p.val = -1;
		last_non_random_p.action = -1;
	}

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
			network->trainer.push(current_experience);							

			has_pending_experience = false;
		}
	}

	Policy last_p, last_non_random_p;
	
	std::string detail() const { return str(format("%s%s")%last_non_random_p.to_string()%(last_p.is_random()? str(format(" *RAND* %d")%last_p.action):"")); }

	int forward(SingleFrameSp frame,std::function<int()> random_action,std::function<bool(int)> is_valid_action)
	{
		forward_passes++;
		
		flush(frame);

		if (forward_passes > temporal_window + 1)
		{
			has_pending_experience = network->epsilon.is_learning;
			std::copy(frame_window.begin(), frame_window.end(), current_experience.input_frames.begin());
			auto p = network->predict(current_experience.input_frames,random_action,is_valid_action);
			last_p = p;
			if (!p.is_random())
			{
				last_non_random_p = p;
			}
			current_experience.action = p.action;

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