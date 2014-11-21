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

DEFINE_bool(gpu, false, "Use GPU to brew Caffe");
DEFINE_int32(iterations,1000000, "iterations");
DEFINE_string(solver, "dqn_solver.prototxt",  "The solver definition protocol buffer text file.");

#include "dqn.h"
#include "game.h"

int main(int argc, char** argv) 
{
	std::mt19937 random_engine;

	caffe::GlobalInit(&argc,&argv);
	// google::InitGoogleLogging(argv[0]);
  	// google::InstallFailureSignalHandler();
 	// google::LogToStderr();

	if (FLAGS_gpu)
	{
		Caffe::set_mode(Caffe::GPU);
	}
	else
	{
		Caffe::set_mode(Caffe::CPU);
	}
	
	Caffe::set_phase(Caffe::TRAIN);

	GameState game_state;

	boost::shared_ptr<DeepNetwork> hero_net(new DeepNetwork);	
	boost::shared_ptr<DeepNetwork> minion_net(new DeepNetwork);	

	std::vector<boost::shared_ptr<DeepNetwork>> nets;
	nets.push_back(hero_net);

	auto train_nets = [&]{
		for (auto n : nets)
		{
			n->train();
		}
	};
	
	for (; game_state.clock<FLAGS_iterations;game_state.epoch++)
	{
		World w(random_engine,game_state);	
		Display disp(w);		

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

		auto hero = [&](int team){return static_cast<Agent*>(new Hero(team));};
		auto minion = [&](int team){return static_cast<Agent*>(new Minion(team));};

		auto spawn = [&](int team,std::function<Agent*(int team)> gen){
			auto pawn = w.spawn([&]{return gen(team);});
			static_cast<Pawn*>(pawn)->brain.reset(new HeroBrain(hero_net,&w));
			pawn->pos = pos_gen([&]{return Vector(x_dist(random_engine),team * (w.size.y - 1));});				
		};			

		spawn(0,minion);
		spawn(0,minion);
		spawn(1,minion);
		spawn(1,minion);
		spawn(0,hero);
		spawn(1,hero);		

		while (!w.quit)
		{
			w.tick();
			train_nets();
			disp.tick();
		}	
	}	
	return 0;
}