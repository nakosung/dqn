auto ANSI = "\033[";

DEFINE_int32(display_interval, 1, "display_interval");
DEFINE_int32(display_after, 2000, "display_after");

bool is_valid_team( int team )
{
	return team == 0 or team == 1;
}

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

	bool is_invalid() const
	{
		return (x < 0 || y < 0 || x >= world_size || y >= world_size);
	}
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

	virtual void game_over(int winner) {}

	virtual void forward() {}
	virtual void tick() 
	{
		check_sanity();
	}
	virtual void backward() {}
	virtual std::string detail() { return ""; }
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
	int& clock;
	std::deque<std::string> events;
	bool quit;	
	int final_winner;
	
	World(std::mt19937& random_engine, int& clock) 
	: random_engine(random_engine), size(world_size,world_size), clock(clock), quit(false), final_winner(-1)
	{				
	}

	void log_event(std::string s)
	{
		events.push_back(s);
		if (events.size() > 10)
			events.pop_front();
	}

	void game_over(int winner)
	{
		final_winner = winner;

		for (auto a : agents)
		{
			if (!a->pending_kill)
			{				
				a->game_over(winner);			
			}
		}
		quit = true;
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
			assert(!a->pos.is_invalid());
		}

		for (auto a : agents)
		{
			a->backward();
		}				

		collect_garbage();
	}

	void collect_garbage()
	{
		for (auto itr = agents.begin(); itr != agents.end();)
		{
			auto agent = *itr;
			if (agent->pending_kill)
			{
				agents.erase(itr++);
			}
			else
			{
				++itr;
			}
		}
	}	

	bool is_vacant(const Vector& x) const
	{
		if (x.is_invalid()) return false;

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
};

class Actable : public Agent
{
public:
	typedef Agent Base;

	float reward;
	boost::shared_ptr<AgentBrain> brain;

	Actable() : num_actions(1), reward(0), brain(nullptr) {}

	virtual std::string detail() { return str(format("%s R(%.2f) B(%s)")%Base::detail()%reward%(brain ? brain->detail() : "none")); }

	int num_actions;
	int action;	

	virtual void forward()
	{
		Base::forward();

		reward = 0;

		if (brain)
		{
			for (;;)
			{
				action = brain->forward(this);
				if (is_valid_action(action)) break;
			}
		}
		else
		{
			action = random_action();
			assert(is_valid_action(action));
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

		if (is_valid_action(action))
		{
			do_action(action);
		}		
	}

	virtual void backward() 
	{
		Base::backward();

		if (brain)
		{
			reward *= 0.1f;			
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

	virtual bool is_valid_action(int action) const { return ::is_valid_action(action); }
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

	virtual bool is_valid_action(int action) const
	{
		if (action < move_action_offset)
			return super::is_valid_action(action);
		
		return can_move(dir_vec[action - move_action_offset]);
	}

	virtual void do_action(int action)
	{
		if (action < move_action_offset)
			return super::do_action(action);

		int move = action - move_action_offset;
		assert(move >= 0 && move < move_max);

		do_move(dir_vec[move]);
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

	std::array<int,2>& scores;			

	Display(World& world,int epoch,int& clock,std::array<int,2>& scores) 
	: world(world), needs_clear(true), epoch(epoch), clock(clock), scores(scores)
	{
	}	

	void tick()
	{
		if (world.quit)
		{			
			if (is_valid_team(world.final_winner))
			{
				scores[world.final_winner] ++;
			}			
		}
		if (world.should_display())
		{
			dump();
			std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+4);
		}		
	}	

	std::array<std::string,20> lines;
	std::string empty;

	void dump()
	{
		if (needs_clear)
		{
			needs_clear = false;
			std::cout << ANSI << "1J";
		}		

		std::list<Agent*> dirties;
		std::vector<std::string> newlines;
		int y = 0;
		auto logline = [&](std::string x) { newlines.push_back(x); };
		logline(str(format("agents %3d clock %8d epoch %8d")%world.agents.size()%clock%epoch));
		logline(str(format("score: %d : %d")%scores[0]%scores[1]));

		// I know, it's too slow.. :)
		std::string reset(str(format("%s40m")%ANSI));
		for (int y=0; y<world.size.y; ++y)
		{
			std::string line = reset + ANSI_ESCAPE::gotoxy(0,y);			

			for (int x=0; x<world.size.x; ++x)
			{
				bool found = false;
				for (auto a : world.agents)
				{
					if (a->pos.x == x && a->pos.y == y)
					{			
						found = true;			
						line += a->one_letter();
						break;
					}								
				}

				if (!found)
				{
					line += " ";					
				}
			}

			std::cout << line;
		}

		for (auto a : world.agents)
		{
			logline(a->detail());
		}
		for (auto a : world.events)
		{
			logline(a);
		}

		for (int line=0; line<lines.size(); ++line)
		{
			const auto& newline = line < newlines.size() ? newlines[line] : empty;
			if (newline != lines[line])
			{
				lines[line] = newline;
				std::cout << ANSI_ESCAPE::gotoxy(world.size.x+5,line+1) << str(format("%-50s")%newline);
			}
		}	

		std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+1) << ANSI << "47;0m";
	}
};

enum PawnType
{
	PT_minion,
	PT_range,
	PT_hero,
	PT_max
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
	PawnType type;
	int range;

	float attack_reward, kill_reward;

	std::string colorize(std::string in) const { return str(format("%s%dm%s%s0m")%ANSI%(team+44)%in%ANSI); }

	virtual std::string one_letter() { return colorize(str(format("%c")%code)); }
	virtual std::string detail() { return colorize(str(format("%c[t:%d] hp:%d cd:%d %s")%code%team%health%cooldown%Base::detail())); }

	Pawn(PawnType type, int team,int in_max_cooldown,int in_max_health, char code, int range, float attack_reward, float kill_reward)
	: type(type), max_health(in_max_health), max_cooldown(in_max_cooldown), team(team), health(in_max_health), cooldown(0), code(code), death_timer(0), range(range), attack_reward(attack_reward), kill_reward(kill_reward)
	{
		num_actions += 1;
	}

	virtual void game_over(int winner)
	{
		if (team == winner)
		{
			reward = 100.0f;
		}
		else
		{
			reward = -100.0f;
		}

		brain->flush(SingleFrameSp());
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

	virtual Pawn* find_target() const
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

	virtual bool is_valid_action(int action) const
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
	Minion(int team) : Pawn(PT_minion,team,5,2,'m',1,0.1f,0.5f) {}
};

class RangeMinion : public Pawn
{
public :
	RangeMinion(int team) : Pawn(PT_range,team,3,1,'r',2,0.1f,0.5f) {}
};

class Hero : public Pawn
{
public :
	Hero(int team) : Pawn(PT_hero, team, 15,3,'H',3,0.2f,1.0f) {}
	virtual void die(Pawn* attacker)
	{
		world->game_over(attacker ? attacker->team : -1);

		brain->flush(SingleFrameSp());
	}
};

class HeroBrain : public AgentBrain
{
public:
	virtual int forward( Actable* agent )
	{
		// LOG(INFO) << "hero brain forward, calling super";
		return Brain::forward(
			get_frame(agent),
			[&]{return agent->random_action();},
			[&](int action){return agent->is_valid_action(action);}
			);
	}

	SingleFrameSp get_frame(Actable* agent) const
	{		
		SingleFrameSp single_frame(new SingleFrame);

		auto& image = single_frame->image;

		std::fill(image.begin(),image.end(),-1);

		Pawn* self = dynamic_cast<Pawn*>(agent);

		auto write = [&](int x, int y, float val)
		{
			Vector v(x - self->pos.x + sight_diameter/2,y - self->pos.y + sight_diameter/2);
			if (v.x >= 0 && v.y >= 0 && v.x < sight_diameter && v.y < sight_diameter)
			{
				image[v.x + v.y * world_size] = val;
			}			
		};

		for (int y=0; y<sight_diameter; ++y)
		{
			for (int x=0; x<sight_diameter; ++x)
			{
				write(x,y,-0.5f);
			}
		}

		for (auto other : agent->world->agents)
		{
			Pawn* a = dynamic_cast<Pawn*>(other.get());
			if (a == nullptr) continue;

			float value = a->health / a->max_health;
			value /= 2;
			value += a->type;
			value /= PT_max * 2;
			value += (a->team == self->team) ? 1 : 0;			
			write(a->pos.x,a->pos.y,value);
		}
		return single_frame;
	}
};