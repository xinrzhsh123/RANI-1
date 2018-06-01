# clearNuclear

software:   ubuntu16.04    theano  python2.7.12   keras   cuda8.0   CuDNN5.1 and etc
hardware:   nvida gpus  https://developer.nvidia.com/cuda-gpus  (my gpu is GeForce GTX 960M)

installation steps: 
************************************************************
## 1) install ubuntu16.04    install python2.7.12 （shipped with ubuntu)

## 2) if  have GPU,install CUDA8.0,download address: https://developer.nvidia.com/cuda-downloads  (cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb, it has been packed with this installation file) open terminal and excute:
```
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
sudo apt update
sudo apt install cuda
```
Add CUDA  path:
```
sudo gedit /etc/bash.bashrc
```
Add in bash.bashrc file:
'''
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo gedit ~/.bashrc
'''
Add same info in .bashrc file
Then reboot system
test if CUDA has been installed successful?
```
nvcc -V 
```
## 3)  install CuDNN5.1

Step 1: Register an nvidia developer account and download cudnn here (about 80 MB :cudnn-8.0-linux-x64-v5.1.tgz) ,(I have packaged many softwares need for using unccell with this installation file)
Step 2: Check where your cuda installation is. For the installation from the repository it is /usr/lib/... and /usr/include. Otherwise, it will be /urs/local/cuda/. You can check it with which nvcc or ldconfig -p | grep cuda
Step 3: Copy the files:
```
cd folder/extracted/contents
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```
### 4)  Run the following commands:
```
sudo apt-get install python-pip
sudo apt-get install python-nose
sudo apt-get install numpy   or pip install numpy 
```
#Test if success,input python in command line, and then:
import numpy
numpy.test()
sudo apt-get install python-scipy  or  pip install scipy （new added）
#Test if success,input python in command line, and then:
```
import scipy
scipy.test()
```
                            If not OK!   Uninstall numpy and scipy
# uninstall numpy
```
sudo apt-get remove python-numpy
```
# uninstall scipy
```
sudo apt-get remove python-scipy
```
# next:
```
sudo apt-get install gfortran 
sudo apt-get install libopenblas-dev 
sudo apt-get install liblapack-dev 
sudo apt-get install libatlas-base-dev
pip install palettable libtiff tifffile 
pip install PyWavelets  mahotas
sudo apt-get install libhdf5-dev
sudo pip install h5py
sudo pip install matplotlib
sudo apt-get install python-tk
sudo pip install scikit-image
sudo pip install scikit-learn
sudo apt-get install python-scipy python-dev python-pip g++ libopenblas-dev git
```
next:
 
install theano (http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu)

```
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
sudo pip install -U --pre pip setuptools wheel
sudo pip install -U --pre theano   (ver 0.9)
sudo apt-get install g++-4.9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
echo -e "\n[nvcc]\nflags=-D_FORCE_INLINES\n" >> ~/.theanorc
```

install keras:
**must install keras 1.2.2**
``
sudo pip install https://github.com/fchollet/keras/tarball/1.2.2
``
Switching from TensorFlow to Theano，By default, Keras will use TensorFlow as its tensor manipulation library. Follow these instructions to configure the Keras backend.
If you have run Keras at least once, you will find the Keras configuration file at:
```
sudo gedit ~${HOME}/.keras/keras.json
```
If it isn't there, you can create it.
The default configuration file looks like this:
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
Simply change the field backend to either "theano" or "tensorflow", and Keras will use the new configuration next time you run any Keras code.
```
gedit ~/.theanorc
```
Add the following info into this file:
```
[global]
openmp=False 
device = gpu
floatX = float32 
allow_input_downcast=True 
[lib]
cnmem = 0.8 
[blas]
ldflags= -lopenblas
[nvcc]
fastmath = True 
```
if you use CPU only， .theanorc will be set as：
```
[global]
openmp=True 
device = cpu 
floatX = float32 
allow_input_downcast=True 
[blas]
ldflags= -lopenblas 
```

## 5) Install matlab Engine in Your Home Folder

To install the engine API for your use only, use the --user option to install to your home folder.
```
cd "matlabroot\extern\engines\python"
python setup.py install --user 
```

## 6) Running the experiment on Raw Data
The code is written in Python, run the "clearNuclear_running.py" to segment the images in folder "original_Nuclear_images"
The final results of fulll method is saved in result_clearNuclear/3_FCCNN













