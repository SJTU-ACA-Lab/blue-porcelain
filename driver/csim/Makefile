SIMX_DIR = ../../sim/csim

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../include -I../common -I../../config -I$(SIMX_DIR)/../common -I$(SIMX_DIR) -I$(CUDA_INSTALL_PATH)/include
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DDUMP_PERF_STATS

CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include



LDFLAGS += -shared -pthread
LDFLAGS += -L. -lcmodel

SRCS = gpgpu.cpp ../common/gpgpu_utils.cpp


# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = libcdriver.so
CMODEL = libcmodel.so

.PHONY: $(CMODEL)

all: $(CMODEL) $(PROJECT)

$(CMODEL):
	DESTDIR=../../driver/csim $(MAKE) -C $(SIMX_DIR) ../../driver/csim/libcmodel.so

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	DESTDIR=../../driver/csim $(MAKE) -C $(SIMX_DIR) clean
	rm -rf libcmodel.so $(PROJECT) *.o
