ifeq ($(CUDA_VERSION_STRING),)
	VERSION = 11.0
else
	VERSION = $(CUDA_VERSION_STRING)
endif 

ifeq ($(DRIVER), csim)
	DRIVER_DIR = ../driver/csim
else ifeq ($(DRIVER), rtlsim)
	DRIVER_DIR = ../driver/rtlsim
else 
	DRIVER_DIR = ../driver/csim
endif

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -Wno-unused-parameter -Wno-return-type -Wno-pointer-arith

CXXFLAGS += -fPIC -Wno-maybe-uninitialized -Wl,-soname,libcudart.so.$(VERSION) -Wl,--version-script=linux-so-version.txt
CXXFLAGS += -I../config -I../driver/include -I$(DRIVER_DIR) -I$(CUDA_INSTALL_PATH)/include

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DRIVER_DIR) -lcdriver

CONFIG = $(CONFIGS)
SRCS = cuda_runtime_api.cc

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

export CONFIG

PROJECT = libcudart.so.$(VERSION)

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(MAKE) -C $(DRIVER_DIR)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
clean:
	rm -rf $(PROJECT)
