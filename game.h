auto ANSI = "\033[";

DEFINE_int32(display_interval, 1, "display_interval");
DEFINE_int32(display_after, 20000, "display_after");

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
	return square(a.x - b.x) + square(a.y - b.y);
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

class World;
class Actable;

class Agent {
public:
	std::function<Agent*()> creator;
	World* world;
	Vector pos;
	bool pending_kill;

	Agent()
	: world(world), pos(0,0), pending_kill(false)
	{}

	virtual void check_sanity() const
	{
		assert(world);
	}

	virtual void forward() {}
	virtual void tick() 
	{
		check_sanity();
	}
	virtual void backward() {}
	virtual std::string detail() { return str(format("pos(%3d,%3d) ")%pos.x%pos.y); }
	virtual std::string one_letter() { return "*"; }
};

class World {
public:
	std::mt19937& random_engine;
	int randint(int N) const
	{
		return std::uniform_int_distribution<>(0,N-1)(random_engine);
	}
	bool should_display() const
	{
		return clock >= FLAGS_display_after && clock % FLAGS_display_interval == 0;
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
					if (agents.size() == 0)
					{
						quit = true;
					}
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
			assert(!is_invalid(a->pos));
		}

		for (auto a : agents)
		{
			a->backward();
		}				
	}

	bool is_invalid(const Vector& x) const
	{
		return (x.x < 0 || x.y < 0 || x.x >= world_size || x.y >= world_size);
	}

	bool is_vacant(const Vector& x) const
	{
		if (is_invalid(x)) return false;

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

class AgentBrain : public Brain
{
public:	
	World* world;
	virtual int forward(Actable* agent ) = 0;
	virtual bool needs_explanation() const { return world->should_display(); }
};

class Actable : public Agent
{
public:
	typedef Agent Base;

	float reward;
	AgentBrain* brain;

	Actable() : num_actions(1), reward(0), brain(nullptr) {}

	virtual std::string detail() { return str(format("%s reward(%f) brain(%s)")%Base::detail()%reward%(brain ? brain->detail() : "none")); }

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

	virtual void check_sanity() const
	{
		Base::check_sanity();

		assert(!std::isnan(reward));
	}

	virtual void tick()
	{
		Base::tick();

		do_action(action);
	}

	virtual void backward() 
	{
		Base::backward();

		if (brain)
		{
			brain->backward(std::min(1.0f,std::max(-1.0f,reward)));
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

static Vector dir_vec[] = {{1,0},{0,1},{-1,0},{0,-1}};

class Movable : public Actable
{
public:
	typedef Actable super;
	int move_action_offset;

	enum {
		move_left,
		move_right,
		move_up,
		move_down,
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
		
		return can_move(dir_vec[action - move_action_offset]);
	}

	virtual void do_action(int action)
	{
		if (action < move_action_offset)
			return super::do_action(action);

		do_move(dir_vec[action - move_action_offset]);
	}

	bool can_move(const Vector& dir) const
	{
		auto new_pos = pos + dir;

		return world->can_move_to(this,pos,new_pos);		
	}

	void do_move(const Vector& dir)
	{
		auto new_pos = pos + dir;

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
		if (world.should_display())
		{
			dump();
			std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+4);
		}		
	}	

	void dump()
	{
		if (needs_clear)
		{
			//needs_clear = false;
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
					if (a->pos.x == x && a->pos.y == y && !a->pending_kill)
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
	int death_timer;
	char code;
	int range;

	float attack_reward, kill_reward;

	std::string colorize(std::string in) const { return str(format("%s%dm%s%s0m")%ANSI%(team+44)%in%ANSI); }

	virtual std::string one_letter() { return colorize(str(format("%c")%code)); }
	virtual std::string detail() { return colorize(str(format("%c[team:%d] %s hp:%d")%code%team%Base::detail()%health)); }

	Pawn(int team,int in_max_cooldown,int in_max_health, char code, int range, float attack_reward, float kill_reward)
	: max_health(in_max_health), max_cooldown(in_max_cooldown), team(team), health(in_max_health), cooldown(0), code(code), death_timer(0), range(range), attack_reward(attack_reward), kill_reward(kill_reward)
	{
		num_actions += 1;
	}

	virtual void check_sanity() const
	{
		Base::check_sanity();

		assert(health <= max_health);
		assert(cooldown <= max_cooldown);
		assert(range >= 0);
		assert(team >= 0 && team < 2);
		assert(!std::isnan(attack_reward));
		assert(!std::isnan(kill_reward));
	}

	virtual Pawn* find_target()
	{
		int best_dist = square(range+1);
		Pawn* best = nullptr;
		for (auto a:world->agents)
		{
			auto b = dynamic_cast<Pawn*>(a.get());
			if (b && b != this && !b->pending_kill && b->team != team)
			{
				auto dist = distance(pos,b->pos);
				if (dist < best_dist)
				{
					// std::cout << "found_target" << dist << pos.x << "," << pos.y << ":" << b->pos.x << b->pos.y;
					best_dist = dist;
					best = b;
				}
			}
		}			
		return best;
	}		

	virtual void die(Pawn* attacker)
	{	
		if (attacker && !attacker->pending_kill)
		{
			attacker->death_timer = 0;
			attacker->reward += kill_reward;
		}		

		pending_kill = true;

		// reward -= 10.0f;
	}

	virtual void take_damage(int damage, Pawn* attacker)
	{	
		health -= damage;
		if (attacker && !attacker->pending_kill)
		{
			attacker->reward += attack_reward;
		}

		// reward -= 1.0f;

		if (health <= 0)
		{		
			health = 0;	
			die(attacker);
		}
	}

	virtual void tick()
	{
		Base::tick();

		death_timer++;
		if (death_timer == 100)
		{
			// take_damage(2,nullptr);
		}

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
	Minion(int team) : Pawn(team,1,1,'m',1,0.1f,0.5f) {}
};

class Hero : public Pawn
{
public :
	Hero(int team) : Pawn(team,1,3,'H',3,0.2f,1.0f) {}
	virtual void die(Pawn* attacker)
	{
		world->quit = true;

		brain->flush(SingleFrameSp());
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