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

DEFINE_int32(gpu, 0, "GPU mode on given device ID");
DEFINE_int32(display_interval, 100, "display_interval");
DEFINE_int32(iterations,1000000, "iterations");
DEFINE_string(solver, "",  "The solver definition protocol buffer text file.");

#include "dqn.h"
#include "game.h"

int main(int argc, char** argv) 
{
	std::mt19937 random_engine;

	caffe::GlobalInit(&argc,&argv);
	// google::InitGoogleLogging(argv[0]);
  	// google::InstallFailureSignalHandler();
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

	boost::shared_ptr<DeepNetwork> hero_net(new DeepNetwork);	
	boost::shared_ptr<DeepNetwork> minion_net(new DeepNetwork);
	
	typedef boost::shared_ptr<AgentBrain> BrainSp;
	enum { num_teams = 2 };
	enum { minions_per_team = 3 };
	enum { num_heroes = 2 };
	enum { num_minions = num_teams * minions_per_team };

	auto gen_brains = [&](int num_brains, boost::shared_ptr<DeepNetwork> net, std::function<AgentBrain*()> gen) {
		std::vector<BrainSp> hero_brains(num_brains);
		std::generate(hero_brains.begin(),hero_brains.end(),[&] { 
			auto brain = BrainSp(gen());
			brain->network = net;
			return brain;
		});
		return hero_brains;
	};

	auto hero_brains = gen_brains(num_heroes,hero_net,[=]{return new HeroBrain;});
	auto minion_brains = gen_brains(num_minions,minion_net,[=]{return new HeroBrain;});	

	int epoch = 0;
	for (int iter = 0; iter<FLAGS_iterations;epoch++)
	{
		World w(random_engine);	
		Display disp(w,epoch,iter);		

		auto pos_gen = [&](std::function<Vector()> gen)
		{
			for (int trial=0;trial<1000;trial++)
			{				
				auto pos = gen();
				if (w.is_vacant(pos)) return pos;
			}

			LOG(FATAL) << "Couldn't find a valid spawn-point";
			exit(-1);
		};

		auto x_dist = std::uniform_int_distribution<>(0,w.size.x-1);

		int minion_id = 0;

		auto spawn_hero = [&](int team){
			Vector pos = pos_gen([&]{return Vector(x_dist(random_engine),team * (w.size.y - 1));});		
			auto brain = hero_brains[team];
			brain->world = &w;
			return w.spawn([=]{
				auto m = new Hero(team);
				m->brain = brain.get();
				m->pos = pos;
				return m;
			});			
		};	
		auto spawn_minion = [&](int team){
			Vector pos = pos_gen([&]{return Vector(x_dist(random_engine),team * (w.size.y - 1));});		
			auto brain = minion_brains[minion_id++];
			brain->world = &w;
			assert(brain.get()!=nullptr);
			return w.spawn([=]{
				auto m = new Minion(team);
				m->brain = brain.get();
				m->pos = pos;
				return m;
			});			
		};	
		// spawn_minion(0);
		// spawn_minion(0);
		// spawn_minion(0);
		// spawn_minion(1);
		// spawn_minion(1);
		// spawn_minion(1);	
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