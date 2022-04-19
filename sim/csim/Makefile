DESTDIR ?= .

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../common -I../../config
CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include
CXXFLAGS += $(CONFIGS)

SRCS = util.cpp mem.cpp 
SRCS +=  core.cpp processor.cpp kernel.cpp threadengine.cpp regfile.cpp  instruction.cpp instr_mov.cpp operand.cpp operandTypeAssigner.cpp instr_miscell.cpp instr_int.cpp instr_ldst.cpp instr_ctrl.cpp instr_fp.cpp instr_pred.cpp simt_stack.cpp
OBJ_DIR := obj_dir
OBJS := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRCS)))
VPATH = ../common/:./

#$(info OBJS is $(OBJS))
#$(info VPATH is $(VPATH))

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else    
	CXXFLAGS += -O3 -DNDEBUG
endif

# XLEN parameterization
ifdef XLEN
	CXXFLAGS += -DXLEN=$(XLEN)
endif

PROJECT = cmodel

# all: $(DESTDIR)/$(PROJECT)
	
# $(DESTDIR)/$(PROJECT): $(SRCS) main.cpp
# 	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(DESTDIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(OBJ_DIR) $(DESTDIR)/$(PROJECT) $(DESTDIR)/lib$(PROJECT).so
