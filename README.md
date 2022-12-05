# Blue-porcelain GPGPU
Blue-porcelain is a GPGPU design proposed by the Advanced Architecture Laboratory of Shanghai Jiao Tong University.
## Directory structure

- `driver`: Host drivers repository.
- `runtime`: Kernel Runtime software.
- `sim`: Simulators repository.
- `benchmarks`: Benchmarks for testing.
- `ci`: Continuous integration scripts.

## Build
### Supported OS Platforms
- Ubuntu 18.04
### Toolchain Dependencies
- gcc 7.5.0
- g++ 7.5.0
- GNU Make 4.1
- CUDA 10.2 Runtime
### Install development tools 

```shell
sudo apt-get install build-essential
sudo apt-get install git bc numdiff
```

### Install codebase

```shell
git clone https://github.com/SJTU-ACALab/blue-porcelain.git
```

### Setup Environment
Make sure that the CUDA_INSTALL_PATH is set to the location (e.g., /usr/local/cuda) where the CUDA Toolkit is installed and the `$CUDA_INSTALL_PATH/bin` is in your PATH. You can add the following instructions to the .bashrc file (assume the CUDA Toolkit is installed in /usr/local/cuda):

```shell
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

### Build sources
```shell
source setup_environment
make -s
```

### Run app
Before running a simulation for an application, make sure you have set up the environment. Then, just simply input the application execution command.

```shell
source setup_environment
./application
```

### Run ci benchmark test
You can run our benchmark automatically under the `ci` directory. Make sure you have set up the environment. Also, the integration test is supported by Python. Please install Python(3.6 ~ 3.9 is available) if you want to use this tool.

    $ source setup_environment
    $ cd ./ci
    $ sh run.sh -a

`-core`: set the sm core number(default is 1)

`-a`   : build both the simulation program and benchmark program

`-b`   : only build the simulation program

`-bb`  : only build the benchmark program

## Docker  
Setup the development environment with Docker.

```shell
cd blue-porcelain
docker build -t acalab/gpgpu:0.1 .
docker run -w /root -it acalab/gpgpu:0.1 /bin/bash
```

## OpenGPU GPGPU Platform
For more information about our Open GPGPU Platform, go here: https://gpgpuarch.org

## License

License information can be found in the LICENSE file. 

Third party license information can be found in the THIRDPARTY file.
