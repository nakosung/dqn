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
DEFINE_int32(iterations,200000, "iterations");
DEFINE_string(solver, "dqn_solver.prototxt",  "The solver definition protocol buffer text file.");
DEFINE_string(model, "", "trained model filename");

#include "dqn.h"
#include "game.h"
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;
 
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
 
  ch = getchar();
 
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
 
  if(ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }
 
  return 0;
}

int main(int argc, char** argv) 
{
	std::mt19937 random_engine;

	caffe::GlobalInit(&argc,&argv);
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

	Environment env(random_engine);

	boost::shared_ptr<DeepNetwork> dqn(new DeepNetwork(env,FLAGS_solver));	
	boost::shared_ptr<DeepNetwork> dqn_trained(new DeepNetwork(env,FLAGS_solver));	
	if (FLAGS_model != "")
	{
		dqn_trained->loader.load_trained(FLAGS_model);
	}
	else
	{
		dqn_trained = dqn;
	}
	
	
	std::vector<boost::shared_ptr<DeepNetwork>> nets;
	nets.push_back(dqn);

	auto train_nets = [&]{
		for (auto n : nets)
		{
			if (n->epsilon.is_learning)
			{
				n->train();				
			}
		}
	};

	bool quit = false;	
	for (;!quit;game_state.epoch++)
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
			static_cast<Pawn*>(pawn)->brain.reset(new HeroBrain(team == 0 ? dqn : dqn_trained,&w));
			pawn->pos = pos_gen([&]{return Vector(x_dist(random_engine),team * (w.size.y - 1));});				
		};			

		spawn(0,minion);
		spawn(0,minion);
		spawn(1,minion);
		spawn(1,minion);
		spawn(0,hero);
		spawn(1,hero);		

		while (!w.quit && !quit)
		{
			if (kbhit())
			{
				switch (auto ch = getchar())
				{
				case 27 : 
					quit = true;
					break;
				case '1' :
				case '2' :
				case '3' : 
				case '4' :
				case '5' :
				case '6' :
					FLAGS_display_interval = 1 << (ch - '1');
					break;				
				}
			}
			w.tick();
			train_nets();
			disp.tick();
		}	
	}	
	return 0;
}