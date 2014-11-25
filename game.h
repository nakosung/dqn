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
	float x, y;

	Vector() {}
	Vector(float x, float y) : x(x), y(y) {}

	bool is_invalid() const
	{
		return (x < 0 || y < 0 || x >= world_size || y >= world_size);
	}
};

Vector operator * (const Vector& a, float k)
{
	return Vector(a.x * k, a.y * k);
}

template <typename T>
T square(T t)
{
	return t*t;
}

float distance_squared( const Vector& a, const Vector& b )
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
		event_die,
		event_heal
	};
	
	EventType type;
	Vector location;

	Event() {}
	Event(EventType type, const Vector& location) : type(type), location(location)
	{}
	std::string one_letter() const 
	{ 
		switch (type)
		{
			case event_attack : return "A";
			case event_takedamage : return "D";
			case event_die : return "X";
			case event_heal : return "+";
			default : return "?";
		}
	}
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
	int geom;
	
	World(std::mt19937& random_engine, GameState& game_state) 
	: random_engine(random_engine), size(world_size,world_size), game_state(game_state), quit(false), final_winner(-1), world_clock(0)
	{		
		geom = std::uniform_int_distribution<>(0,1)(random_engine);		
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
		events.clear();

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

	bool is_vacant(const Vector& x,const Agent* self = nullptr) const
	{
		if (is_solid(x)) return false;

		for (auto a : agents)
		{
			if (a.get() != self && distance_squared(a->pos,x) <= 1)
			{
				return false;
			}
		}
		return true;
	}

	bool can_move_to(const Agent* a,const Vector& start, const Vector& end) const
	{
		return (is_vacant(end,a) && end.x >= 0 && end.y >= 0 && end.x < size.x && end.y < size.y);		
	}

	bool is_solid(const Vector& v) const
	{	
		if (v.is_invalid()) return true;

		switch (geom)
		{
		case 0 : return false;
		case 1 : return false;//abs(v.x - world_size/2) <= world_size / 4 && abs(v.y - world_size/2) <= 0;
		default : return false;
		}			
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

static Vector dir_vec[num_move_dirs] = {{1,0},{0,1},{-1,0},{0,-1}};

class Movable : public Actable
{
public:
	typedef Actable super;
	int move_action_offset;
	float speed;

	enum {
		move_left,
		move_right,
		move_up,
		move_down,
		move_max
	};

	Movable()
	: speed(0.125f)
	{
		move_action_offset = num_actions;
		num_actions += move_max;
	}

	virtual bool is_valid_action(int action) const
	{
		if (action < move_action_offset)
			return super::is_valid_action(action);
		
		return can_move(dir_vec[action - move_action_offset] * speed);
	}

	virtual void do_action(int action)
	{
		if (action < move_action_offset)
			return super::do_action(action);

		int move = action - move_action_offset;
		assert(move >= 0 && move < move_max);

		do_move(dir_vec[move] * speed);
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

		int zoom = 2;

		std::list<Agent*> dirties;
		std::vector<std::string> newlines;
		int y = 0;
		auto logline = [&](std::string x) { newlines.push_back(x); };
		logline(str(format("agents %3d clock %8d epoch %8d")%world.agents.size()%world.game_state.clock%world.game_state.epoch));
		logline(str(format("%s(%d) : %s(%d)")%world.game_state.names[0]%world.game_state.scores[0]%world.game_state.names[1]%world.game_state.scores[1]));

		// I know, it's too slow.. :)
		std::string reset(str(format("%s40m")%ANSI));
		for (int y=0; y<world.size.y * zoom; ++y)
		{			
			std::string line = reset + ANSI_ESCAPE::gotoxy(0,y);			

			for (int x=0; x<world.size.x * zoom; ++x)
			{
				auto p = Vector((float)x/zoom,(float)y/zoom);

				if (world.is_solid(p))
				{
					line += "#";
					continue;
				}
				for (bool found = false;;)
				{
					for (const auto& a : world.events)
					{
						if (distance_squared(p, a.location) < 0.25)
						{			
							found = true;			
							line += a.one_letter();
							break;
						}								
					}

					if (found) break;

					for (auto a : world.agents)
					{
						if (distance_squared(p, a->pos) < 0.25)
						{			
							found = true;			
							line += a->one_letter();
							break;
						}								
					}

					if (found) break;

					line += " ";	

					break;
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
				std::cout << ANSI_ESCAPE::gotoxy(world.size.x*zoom+5,line+1) << str(format("%-50s")%newline);
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

enum SkillEffect
{
	SE_nothing,
	SE_heal,
	SE_deal
};

struct SkillParams
{
	SkillEffect type;
	int cooldown;
	int range;
};

class Pawn : public Movable
{
public :
	typedef Movable Base;

	// schema
	int max_health;

	int team;
	int health;	
	int death_timer;
	char code;
	PawnType type;
	int range;

	std::array<SkillParams,max_skills> skill_params;
	std::array<int,max_skills> cooldown;

	float attack_reward, kill_reward;

	std::string colorize(std::string in) const { return str(format("%s%dm%s%s0m")%ANSI%(team+44)%in%ANSI); }

	virtual bool is_friendly(int team) const { return team == this->team; }

	virtual std::string one_letter() { return colorize(str(format("%c")%code)); }
	virtual std::string detail() { return colorize(str(format("%c[%d] hp:%d %s")%code%team%health%Base::detail())); }

	Pawn(PawnType type, int team,const std::array<SkillParams,max_skills>& skill_params,int in_max_health, char code, float attack_reward, float kill_reward)
	: type(type), max_health(in_max_health), skill_params(skill_params), team(team), health(in_max_health), code(code), death_timer(0), attack_reward(attack_reward), kill_reward(kill_reward)
	{
		std::fill(cooldown.begin(),cooldown.end(),0);
		num_actions += max_skills;
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
		assert(team >= 0 && team < 2);
		assert(!std::isnan(attack_reward));
		assert(!std::isnan(kill_reward));
	}

	bool can_affect( SkillEffect type, Pawn* b ) const
	{
		if (b && b != this && !b->pending_kill)
		{
			switch (type)
			{
				case SE_deal : return b->team != team;
				case SE_heal : return b->team == team && b->health < b->max_health;
				default : return false;
			}			
		}
		return false;
	}

	virtual void do_affect(SkillEffect type,Pawn* b)
	{	
		if (can_affect(type,b))
		{
			switch (type)
			{
				case SE_deal: 
					b->take_damage(1,this);
					break;
				case SE_heal:
					b->heal(1,this);
					break;
			}
		}
	}
	

	virtual Pawn* find_target(int slot) const
	{
		const auto& param = skill_params[slot];
		int best_dist = square(param.range+1);
		Pawn* best = nullptr;
		for (auto a:world->agents)
		{
			auto b = dynamic_cast<Pawn*>(a.get());
			if (can_affect(param.type,b))
			{
				auto dist = distance_squared(pos,b->pos);
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
			world->add_event(Event(Event::event_attack,attacker->pos));	
		}

		world->add_event(Event(Event::event_takedamage,pos));

		// reward -= 1.0f;

		if (health <= 0)
		{		
			world->add_event(Event(Event::event_die,pos));	
			health = 0;	
			die(attacker);
		}
	}

	virtual void heal(int amount, Pawn* healer)
	{	
		health += amount;

		if (health > max_health)
		{
			health = max_health;
		}
	}

	float smell() const
	{
		float r = 0.0f;

		for (auto agent : world->agents)
		{
			auto pawn = dynamic_cast<Pawn*>(agent.get());
			if (pawn && pawn != this)
			{
				r += exp( -distance_squared(pos,pawn->pos) / square(range) );
			}
		}
		return r;
	}	

	float skill_pct(int slot) const
	{
		return (float)cooldown[slot] / skill_params[slot].cooldown;
	}

	virtual void tick()
	{
		Base::tick();

		death_timer++;
		if (death_timer == 100)
		{
			// take_damage(2,nullptr);
		}

		for (auto& c : cooldown)
		{
			if (c>0)
				c--;			
		}


		//reward = std::max( reward, smell() * 0.01f );
	}

	virtual bool is_valid_action(int action) const
	{
		if (action < max_skills)
		{	
			return cooldown[action] == 0 && find_target(action) != nullptr;
		}
		else
		{
			return Base::is_valid_action(action-max_skills);
		}
	}

	virtual void do_action(int action)
	{
		if (action < max_skills)		
		{			
			const auto& param = skill_params[action];
			cooldown[action] = param.cooldown;
			auto target = find_target(action);			
			if (target) 
			{
				do_affect(param.type,target);
			}
		}
		else
		{
			Base::do_action(action-max_skills);
		}
	}
};

class Minion : public Pawn
{
public :
	Minion(int team) : Pawn(PT_minion,team,{{SE_deal,5,1}},2,'m',0.1f,0.5f) {}
};

class Hero : public Pawn
{
public :
	Hero(int team) : Pawn(PT_hero, team, {{SE_deal,15,3}},3,'H',0.2f,1.0f) {}
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

		auto write_i = [&](int ch, int x, int y, float val)
		{
			Vector v(x - self->pos.x + sight_diameter/2,y - self->pos.y + sight_diameter/2);
			if (v.x >= 0 && v.y >= 0 && v.x < sight_diameter && v.y < sight_diameter)
			{
				images[ch][v.x + v.y * world_size] += val;
			}			
		};

		auto write = [&](int ch, const Vector& p, float val)
		{
			auto ix = int(std::floor(p.x));
			auto iy = int(std::floor(p.y));
			auto fx = p.x - ix;
			auto fy = p.y - iy;			

			write_i(ch,ix,iy,val * (1-fx) * (1-fy));
			write_i(ch,ix+1,iy,val * fx * (1-fy));
			write_i(ch,ix+1,iy+1,val * fx * fy);
			write_i(ch,ix,iy+1,val * (1-fx) * fy);
		};

		for (int y=0; y<world_size; ++y)
		{
			for (int x=0; x<world_size; ++x)
			{
				if (agent->world->is_solid(Vector(x,y)))
				{
					write_i(0,x,y,-2);
				}				
				else
				{
					write_i(0,x,y,-1);
				}
			}
		}

		for (auto other : agent->world->agents)
		{
			Pawn* a = dynamic_cast<Pawn*>(other.get());
			if (a == nullptr) continue;
									
			write(0,a->pos,a->type);
			write(1,a->pos,(float)a->health / a->max_health);
			write(2,a->pos,(a->team == self->team) ? 1 : -1);			
			for (int i=0; i<max_skills; ++i)
			{
				write(4+i,a->pos,a->skill_pct(i));
			}
		}

		for (const auto& e : agent->world->events)
		{
			write(3,e.location,e.type + 1);
		}

		auto& stats = single_frame->stats;
		stats[0] = world->game_state.clock / 1000.0f;
		stats[1] = (float)self->health / self->max_health;		
		stats[2] = self->type;
		for (int i=0; i<max_skills; ++i)
		{
			stats[3+i] = self->skill_pct(i);
		}		
		return single_frame;
	}
};