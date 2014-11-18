auto ANSI = "\033[";

struct ANSI_ESCAPE
{
	static std::string gotoxy(int x, int y)
	{
		return str(format("%s%d;%dH")%ANSI%(y+1)%(x+1));
	}
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
	return a.x == b.x && a.y == b.y;
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
	std::mt19937& random_engine;
	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}

	Vector size;
	std::list< shared_ptr<Agent> > agents;	
	int clock;
	std::deque<std::string> events;
	bool quit;
	
	World(std::mt19937& random_engine) 
	: random_engine(random_engine), size(world_size,world_size), clock(0), quit(false)
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

	bool is_vacant(const Vector& x) const
	{
		for (auto a : agents)
		{
			if (a->pos == x)
			{
				return false;
			}
		}
		return true;
	}

	bool can_move_to(const Agent* a,const Vector& start, const Vector& end)
	{
		return end.x >= 0 && end.y >= 0 && end.x < size.x && end.y < size.y && is_vacant(end);
	}
};

class Actable : public Agent
{
public:
	typedef Agent Base;

	float reward;
	AgentBrain* brain;

	Actable() : num_actions(1), reward(0), brain(nullptr) {}

	virtual std::string detail() { return str(format("%s reward(%f)")%Base::detail()%reward); }

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
			//LOG(INFO) << "moved!" << this << " : " << new_pos.x << "," << new_pos.y << " <- " << pos.x << ", " << pos.y;
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
			std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+4);
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

		reward -= 10.0f;
	}

	virtual void take_damage(int damage, Pawn* attacker)
	{	
		health -= damage;
		attacker->reward += 1.0f;
		reward -= 1.0f;
		if (health < 0)
		{			
			attacker->reward += 10.0f;

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

		brain->flush(SingleFrameSp());
	}

	virtual Pawn* find_target()
	{
		int best_dist = square(2);
		Pawn* best = nullptr;
		for (auto a:world->agents)
		{
			auto b = dynamic_cast<Pawn*>(a.get());
			if (b && b != this && b->team != team)
			{
				auto dist = distance(a->pos,b->pos);
				if (dist < best_dist)
				{
					// LOG(INFO) << "found_target" << dist << a->pos.x << "," << a->pos.y << ":" << b->pos.x << b->pos.y;
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
		// LOG(INFO) << "hero brain forward, calling super";
		return Brain::forward(get_frame(agent),[&]{return agent->random_action();});
	}

	SingleFrameSp get_frame(Actable* agent) const
	{		
		SingleFrameSp single_frame(new SingleFrame);

		std::fill(single_frame->begin(),single_frame->end(),0.5f);

		Pawn* self = dynamic_cast<Pawn*>(agent);

		auto write = [&](int x, int y, float val)
		{
			(*single_frame)[x + y * world_size] = val;
		};
		for (auto other : agent->world->agents)
		{
			Pawn* a = dynamic_cast<Pawn*>(other.get());
			if (a == nullptr) continue;

			float value = 0.0f;

			if (a == agent)
			{
				value = 1.0f;
			}
			else if (a->team == self->team)
			{
				value = 0.75f;
			}
			write(a->pos.x,a->pos.y,value);
		}
		return single_frame;
	}
};