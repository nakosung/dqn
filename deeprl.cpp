#include "caffe/caffe.hpp"
#include <list>
#include <boost/format.hpp>
#include <random>

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using caffe::Blob;
using boost::str;
using boost::format;

DEFINE_int32(gpu, -1, "GPU mode on given device ID");
DEFINE_int32(display_interval, 100, "display_interval");
DEFINE_int32(iterations,1000000, "iterations");
DEFINE_string(solver, "",  "The solver definition protocol buffer text file.");

auto ANSI = "\033[";

struct ANSI_ESCAPE
{
	static std::string gotoxy(int x, int y)
	{
		return str(format("%s%d;%dH")%ANSI%(y+1)%(x+1));
	}
};


// fix
enum { batch_size = 32 };
enum { temporal_window = 1 };	
enum { world_size = 16 };

// var per
enum { max_actions = 5 };
enum { num_states = world_size * world_size * 3 };
enum { num_actions = max_actions };
enum { net_inputs = (num_states + num_actions) * temporal_window + num_states };

enum { MinibatchSize = batch_size };
enum { InputDataSize = num_states };
enum { MinibatchDataSize = InputDataSize * MinibatchSize };
enum { OutputCount = num_actions };

typedef std::vector<float> input_type;
typedef std::vector<float> net_input_type;
typedef std::vector<float> State;

struct Frame
{
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

	Policy policy(State s) const
	{
		// 
		return Policy{0,1};
	}

	bool test_epsilon() const
	{
		float dice = std::uniform_real_distribution<float>(0,1)(random_engine);
		return dice < epsilon.get();
	}

	int predict(State s,std::function<int()> random_action) const
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
			for (int k=0; k<batch_size; ++k)
			{
				int re = randint(experience_size);
				const Experience& e = experiences[re];				
				auto p = policy(e.next_state);
				float r = e.reward + gamma * p.val;
			}
		}
	}
};

class Brain
{
public:
	typedef boost::shared_ptr<DeepNetwork> NetworkSp;

	int forward_passes;
	float latest_reward;
	
	net_input_type last_input_array;

	NetworkSp network;

	Brain(NetworkSp network)
	: forward_passes(0), latest_reward(0), network(network)
	{
	}

	int forward(const input_type& input_array,std::function<int()> random_action)
	{
		forward_passes++;
		last_input_array = input_array;

		Frame frame;

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

	void append(net_input_type& a, const net_input_type& b)
	{
		a.insert(a.end(),b.begin(),b.end());
	}

	net_input_type get_net_input(const input_type& xt)
	{
		net_input_type w;
		append(w,xt);

		for (int k=0; k<temporal_window; ++k)
		{
			append(w,frame_window[k].state);
			append(w,action1ofk(num_actions,frame_window[k].action));
		}

		return w;
	}	

	std::deque<Frame> frame_window;
};

struct Vector
{
	int x, y;

	Vector() {}
	Vector(int x, int y) : x(x), y(y) {}
};

template <typename T>
T square(T t)
{
	return t*t;
}

int distance( const Vector& a, const Vector& b )
{
	return square(a.x - a.y) + square(b.x - b.y);
}

Vector operator + (const Vector& a, const Vector& b)
{
	return Vector(a.x + b.x, a.y + b.y);
}

bool operator != (const Vector& a, const Vector& b)
{
	return a.x != b.x || a.y != b.y;
}

bool operator == (const Vector& a, const Vector& b)
{
	return a.x == b.x || a.y == b.y;
}

static Vector dir_vec[] = {{1,0},{0,1},{-1,0},{0,-1}};

class World;
class Actable;
class AgentBrain : public Brain
{
public:
	virtual int forward(Actable* agent ) = 0;
};

class Agent {
public:
	std::function<Agent*()> creator;
	World* world;
	Vector pos;
	int dir;
	bool pending_kill;
	Vector saved;	

	Agent()
	: world(world), pos(0,0), dir(0), saved(0,0), pending_kill(false)
	{}

	virtual void forward() {}
	virtual void tick() {}
	virtual void backward() {}
	virtual std::string detail() { return str(format("agent %3d,%3d %d")%pos.x%pos.y%dir); }
	virtual std::string one_letter() { return "*"; }
};

class World {
public:
	mutable std::mt19937 random_engine;
	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}

	Vector size;
	std::list< shared_ptr<Agent> > agents;	
	int clock;
	std::deque<std::string> events;
	bool quit;
	
	World() 
	: size(world_size,world_size), clock(0), quit(false)
	{		
	}

	void log_event(std::string s)
	{
		events.push_back(s);
		if (events.size() > 10)
			events.pop_front();
	}

	Agent* spawn(std::function<Agent*(void)> l)
	{
		auto agent = l();
		agent->world = this;
		agent->creator = l;
		agents.push_back( shared_ptr<Agent>(agent) );

		return agent;
	}

	void tick() 
	{
		clock++;

		while (true)
		{
			bool clear_to_go = true;
			for (auto a : agents)
			{
				if (a->pending_kill)
				{
					if (a->creator)
					{
						spawn(a->creator);
					}
					agents.remove(a);
					clear_to_go = false;
					break;
				}
			}
			if (clear_to_go) break;
		}
		

		for (auto a : agents)
		{
			a->forward();
		}

		for (auto a : agents)
		{
			a->tick();
		}

		for (auto a : agents)
		{
			a->backward();
		}				
	}

	bool can_move_to(const Agent* a,const Vector& start, const Vector& end)
	{
		return end.x >= 0 && end.y >= 0 && end.x < size.x && end.y < size.y;
	}
};

class Actable : public Agent
{
public:
	typedef Agent Base;

	float reward;
	AgentBrain* brain;

	Actable() : num_actions(1), reward(0), brain(nullptr) {}

	int num_actions;
	int action;	

	virtual void forward()
	{
		Base::forward();

		reward = 0;

		if (brain)
		{
			action = brain->forward(this);
		}
		else
		{
			action = random_action();
		}
	}

	virtual void tick()
	{
		do_action(action);
	}

	virtual void backward() 
	{
		Base::backward();

		if (brain)
		{
			brain->backward(reward);
		}
	}

	virtual int random_action() 
	{
		for (;;)
		{
			action = world->randint(num_actions);
			if (is_valid_action(action))
			{
				return action;
			}
		}
	}

	virtual bool is_valid_action(int action) { return true; }
	virtual void do_action(int action) {}
};

class Movable : public Actable
{
public:
	typedef Actable super;
	int move_action_offset;

	enum {
		move_front,
		turn_left,
		turn_right,
		move_max
	};

	Movable()
	{
		move_action_offset = num_actions;
		num_actions += move_max;
	}

	virtual bool is_valid_action(int action)
	{
		if (action < move_action_offset)
			return super::is_valid_action(action);
		switch (action - move_action_offset)
		{
		case move_front :
			return can_move_front();
		default:
			return true;
		}
	}

	virtual void do_action(int action)
	{
		if (action < move_action_offset)
			return super::do_action(action);

		switch (action - move_action_offset)
		{
		case move_front :
			do_move_front();
			break;
		case turn_right :
			dir = (dir + 1 + 4) % 4;
			break;
		case turn_left :
			dir = (dir - 1 + 4) % 4;
			break;
		default:
			// never reaches here!
			break;
		}
	}

	bool can_move_front() const
	{
		auto new_pos = pos + dir_vec[dir];

		return world->can_move_to(this,pos,new_pos);		
	}

	void do_move_front()
	{
		auto new_pos = pos + dir_vec[dir];

		if (world->can_move_to(this,pos,new_pos))
		{
			// LOG(INFO) << "moved!" << new_pos.x << "," << new_pos.y;
			pos = new_pos;
		}
	}
};

class Display
{
public:
	World& world;

	bool needs_clear;
	int& clock;
	int epoch;

	Display(World& world,int epoch,int& clock) 
	: world(world), needs_clear(true), epoch(epoch), clock(clock) 
	{}

	void tick()
	{
		if (world.clock % FLAGS_display_interval == 0)
		{
			dump();
		}
	}

	void dump()
	{
		if (needs_clear)
		{
			needs_clear = false;
			std::cout << ANSI << "1J";
		}		

		std::list<Agent*> dirties;
		int y = 0;
		auto logline = [&](std::string x) { std::cout << ANSI_ESCAPE::gotoxy(world.size.x+5,++y) << str(format("%-50s")%x); };
		logline(str(format("agents %3d clock %8d epoch %8d")%world.agents.size()%clock%epoch));

		// I know, it's too slow.. :)
		for (int y=0; y<world.size.y; ++y)
		{
			std::string reset(str(format("%s40m")%ANSI));
			std::cout << ANSI_ESCAPE::gotoxy(0,y) << reset;		

			bool found = false;
			for (int x=0; x<world.size.x; ++x)
			{
				for (auto a : world.agents)
				{
					if (a->pos.x == x && a->pos.y == y)
					{			
						found = true;			
						std::cout << a->one_letter();
						break;
					}								
				}

				if (!found)
				{
					std::cout << " ";					
				}
			}

		}
		for (auto a : world.agents)
		{
			if (!a->pending_kill)
			{
				logline(a->detail());
			}			
		}
		for (auto a : world.events)
		{
			logline(a);
		}
		while (y<20)
		{
			logline("");
		}		
		
		std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+1) << ANSI << "47;0m";
	}
};

class Pawn : public Movable
{
public :
	typedef Movable Base;

	// schema
	int max_cooldown;
	int max_health;

	int team;
	int health;	
	int cooldown;
	char code;

	std::string colorize(std::string in) const { return str(format("%s%dm%s%s0m")%ANSI%(team+44)%in%ANSI); }

	virtual std::string one_letter() { return colorize(str(format("%c")%code)); }
	virtual std::string detail() { return colorize(str(format("%c[team:%d] %s hp:%d")%code%team%Base::detail()%health)); }

	Pawn(int team,int in_max_cooldown,int in_max_health, char code)
	: max_health(in_max_health), max_cooldown(in_max_cooldown), team(team), health(in_max_health), cooldown(0), code(code)
	{
		num_actions += 1;
	}

	virtual Pawn* find_target()
	{
		auto target = pos + dir_vec[dir];
		for (auto a:world->agents)
		{
			auto b = dynamic_cast<Pawn*>(a.get());
			if (b && b->pos == target && b->team != team)
			{
				return b;
			}
		}
		return nullptr;
	}	

	virtual void die(Pawn* attacker)
	{	
		pending_kill = true;
	}

	virtual void take_damage(int damage, Pawn* attacker)
	{		
		health -= damage;
		if (health < 0)
		{			
			health = 0;	
			die(attacker);
		}
	}

	virtual void tick()
	{
		Base::tick();

		if (cooldown>0)
			cooldown--;
	}

	virtual bool is_valid_action(int action)
	{
		if (action == 0)
		{			
			return cooldown == 0 && find_target() != nullptr;
		}
		else
		{
			return Base::is_valid_action(action-1);
		}
	}

	virtual void do_action(int action)
	{
		if (action == 0)
		{
			cooldown = max_cooldown;
			auto target = find_target();			
			if (target) 
				target->take_damage(1,this);
		}
		else
		{
			Base::do_action(action-1);
		}
	}
};

class Minion : public Pawn
{
public :
	Minion(int team) : Pawn(team,3,1,'m') {}
};

class Hero : public Pawn
{
public :
	Hero(int team) : Pawn(team,10,5,'H') {}
	virtual void die(Pawn* attacker)
	{
		world->quit = true;
	}

	virtual Pawn* find_target()
	{
		int best_dist = world->size.x * world->size.y * 4;
		Pawn* best = nullptr;
		for (auto a:world->agents)
		{
			auto b = dynamic_cast<Pawn*>(a.get());
			if (b && b->team != team)
			{
				auto dist = distance(a->pos,b->pos);
				if (dist < best_dist)
				{
					best_dist = dist;
					best = b;
				}
			}
		}
		return best;
	}	
};

class HeroBrain : public AgentBrain
{
public:
	virtual int forward( Actable* agent )
	{
		return agent->random_action();
	}
};

int main(int argc, char** argv) 
{
	caffe::GlobalInit(&argc,&argv);
	// google::InitGoogleLogging(argv[0]);
 //  	google::InstallFailureSignalHandler();
  	google::LogToStderr();

	if (FLAGS_gpu)
	{
		Caffe::set_mode(Caffe::GPU);
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
	}
	
	Caffe::set_phase(Caffe::TRAIN);

	boost::shared_ptr<DeepNetwork> network(new DeepNetwork);

	int epoch = 0;
	for (int iter = 0; iter<FLAGS_iterations;epoch++)
	{
		World w;	
		Display disp(w,epoch,iter);		

		auto spawn_hero = [&](int team){
			return w.spawn([=]{
				auto m = new Hero(team);
				m->pos.x = w.size.x / 2;
				m->pos.y = team * (w.size.y - 1);
				return m;
			});			
		};	
		auto spawn_minion = [&](int team){
			return w.spawn([=]{
				auto m = new Minion(team);
				m->pos.x = w.size.x / 2;
				m->pos.y = team * (w.size.y - 1);
				return m;
			});			
		};	
		spawn_minion(0);
		spawn_minion(0);
		spawn_minion(0);
		spawn_minion(1);
		spawn_minion(1);
		spawn_minion(1);	
		spawn_hero(0);
		spawn_hero(1);

		for (; !w.quit; ++iter)
		{
			w.tick();
			disp.tick();
		}	
	}	
	return 0;
}