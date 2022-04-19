TEST_PATH=polybench basic

PROJECTS ?=$(shell find $(TEST_PATH) -name Makefile)

FILTER_OUT :=

PROJECTS := $(filter-out $(FILTER_OUT),$(PROJECTS))
%.ph_build :
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)
%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building tests"

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
