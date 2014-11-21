dqn
===

borrowed many things from https://github.com/muupan/dqn-in-the-caffe

```
git clone https://github.com/nakosung/caffe.git
git checkout dqn

cmake . -DCMAKE_CXX_COMPILER=$(which g++)
```

for host machine
```
export DOCKER_NVIDIA_DEVICES="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"
sudo docker run -v /home:/home -ti $DOCKER_NVIDIA_DEVICES nakosung/caffe-gpu /bin/bash
cuda-samples/.../deviceQuery # this is required to make /dev/nvidia-uvm available
```

within docker instance
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64     
```