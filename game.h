auto ANSI = "\033[";

struct GameState
{
	std::array<int,2> scores;
	std::array<std::string,2> names;
	int epoch;
	int clock;

	GameState() 
	: epoch(0), clock(0)
	{
		std::fill(scores.begin(),scores.end(),0);
	}

	void swap_team()
	{
		std::swap(scores[0],scores[1]);
		std::swap(names[0],names[1]);
	}
};

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
	World* world;
	Vector pos;
	bool pending_kill;

	Agent()
	: world(world), pos(0,0), pending_kill(false)
	{}

	virtual bool is_friendly(int team) const { return false; }

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

struct Event
{
	enum EventType
	{
		event_attack,
		event_takedamage,
		event_die
	};
	
	EventType type;
	Vector location;

	Event() {}
	Event(EventType type, const Vector& location) : type(type), location(location)
	{}
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
		return game_state.clock >= FLAGS_display_after && game_state.clock % FLAGS_display_interval == 0;
	}

	Vector size;
	std::list< shared_ptr<Agent> > agents;	
	GameState& game_state;	
	bool quit;	
	int final_winner;
	int world_clock;
	std::vector<Event> events;
	
	World(std::mt19937& random_engine, GameState& game_state) 
	: random_engine(random_engine), size(world_size,world_size), game_state(game_state), quit(false), final_winner(-1), world_clock(0)
	{				
	}

	void add_event(const Event& event)
	{
		events.push_back(event);
	}	

	void game_over(int winner)
	{
		final_winner = winner;

		if (is_valid_team(winner))
		{
			game_state.scores[winner]++;
		}			

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
		agents.push_back( shared_ptr<Agent>(agent) );

		return agent;
	}

	int get_dominant_team() const
	{
		int powers[2] = {0,0};
		for (auto a : agents)
		{
			for (int team=0; team<2; ++team)
			{
				if (a->is_friendly(team))
				{
					powers[team]++;
				}
			}				
		}			

		if (powers[0] > powers[1])
		{
			return 0;
		}
		else if (powers[0] < powers[1])
		{
			return 1;
		}
		else 
		{
			return -1;
		}
	}

	void tick() 
	{
		events.empty();

		game_state.clock++;
		if (world_clock++ > 1000)
		{		
			game_over(get_dominant_team());
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

	bool can_move_to(const Agent* a,const Vector& start, const Vector& end) const
	{
		return !is_solid(end) && end.x >= 0 && end.y >= 0 && end.x < size.x && end.y < size.y && is_vacant(end);
	}

	bool is_solid(const Vector& v) const
	{
		return abs(v.x - world_size/2) <= world_size / 4 && abs(v.y - world_size/2) <= 0;
	}
};

class AgentBrain : public Brain
{
public:	
	AgentBrain(NetworkSp network,World* world) : Brain(network), world(world) {}
	World* world;
	virtual int forward(Actable* agent ) = 0;	
};

class Actable : public Agent
{
public:
	typedef Agent Base;

	float reward, acc_reward;
	boost::shared_ptr<AgentBrain> brain;

	Actable() : num_actions(1), reward(0), brain(nullptr), acc_reward(0) {}

	virtual std::string detail() { return str(format("%s R(%5.2f) B(%s)")%Base::detail()%acc_reward%(brain ? brain->detail() : "none")); }

	int num_actions;
	int action;	

	virtual void forward()
	{
		Base::forward();

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

			acc_reward = acc_reward * brain->network->trainer.gamma + reward;
			reward = 0;
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

	Display(World& world) 
	: world(world), needs_clear(true)
	{
	}	

	void tick()
	{		
		if (world.should_display())
		{
			dump();
			std::cout << ANSI_ESCAPE::gotoxy(0,world.size.y+4);
		}
		else if (world.game_state.clock < FLAGS_display_after && world.game_state.clock % 1000 == 0)
		{
			std::cout << "clock:" << world.game_state.clock << "\n";			
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
		logline(str(format("agents %3d clock %8d epoch %8d")%world.agents.size()%world.game_state.clock%world.game_state.epoch));
		logline(str(format("%s(%d) : %s(%d)")%world.game_state.names[0]%world.game_state.scores[0]%world.game_state.names[1]%world.game_state.scores[1]));

		// I know, it's too slow.. :)
		std::string reset(str(format("%s40m")%ANSI));
		for (int y=0; y<world.size.y; ++y)
		{
			std::string line = reset + ANSI_ESCAPE::gotoxy(0,y);			

			for (int x=0; x<world.size.x; ++x)
			{
				if (world.is_solid(Vector(x,y)))
				{
					line += "#";
					continue;
				}
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

	virtual bool is_friendly(int team) const { return team == this->team; }

	virtual std::string one_letter() { return colorize(str(format("%c")%code)); }
	virtual std::string detail() { return colorize(str(format("%c[%d] hp:%d %s")%code%team%health%Base::detail())); }

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
			world->add_event(Event(Event::event_takedamage,pos));	
			world->add_event(Event(Event::event_attack,attacker->pos));	
		}

		// reward -= 1.0f;

		if (health <= 0)
		{		
			world->add_event(Event(Event::event_die,pos));	
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
	HeroBrain(NetworkSp network, World* world) : AgentBrain(network,world) {}
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

		auto& images = single_frame->images;

		for (auto& image : images)
		{
			std::fill(image.begin(),image.end(),0);
		}		

		Pawn* self = dynamic_cast<Pawn*>(agent);

		auto write = [&](int ch, int x, int y, float val)
		{
			Vector v(x - self->pos.x + sight_diameter/2,y - self->pos.y + sight_diameter/2);
			if (v.x >= 0 && v.y >= 0 && v.x < sight_diameter && v.y < sight_diameter)
			{
				images[ch][v.x + v.y * world_size] = val;
			}			
		};

		for (int y=0; y<world_size; ++y)
		{
			for (int x=0; x<world_size; ++x)
			{
				if (agent->world->is_solid(Vector(x,y)))
				{
					write(0,x,y,-2);
				}				
				else
				{
					write(0,x,y,-1);
				}
			}
		}

		for (auto other : agent->world->agents)
		{
			Pawn* a = dynamic_cast<Pawn*>(other.get());
			if (a == nullptr) continue;
									
			write(0,a->pos.x,a->pos.y,a->type);
			write(1,a->pos.x,a->pos.y,(float)a->health / a->max_health);
			write(2,a->pos.x,a->pos.y,(a->team == self->team) ? 1 : -1);			
		}

		for (const auto& e : agent->world->events)
		{
			write(3,e.location.x,e.location.y,e.type + 1);
		}

		auto& stats = single_frame->stats;
		stats[0] = world->game_state.clock / 1000.0f;
		stats[1] = (float)self->health / self->max_health;
		stats[2] = (float)self->cooldown / self->max_cooldown;
		stats[3] = self->type;
		return single_frame;
	}
};