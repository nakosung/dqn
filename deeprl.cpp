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

#include "dqn.h"
#include "game.h"

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