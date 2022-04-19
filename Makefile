all:
	$(MAKE) -C driver
	$(MAKE) -C runtime

csim:
	$(MAKE) -C driver csim

clean:
	$(MAKE) -C driver clean
	$(MAKE) -C runtime clean
