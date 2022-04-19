NVCC_OPTS=-O3 -arch=sm_70 -Xcompiler -Wall -Xcompiler -Wextra -m64 --cudart shared

PREFIXSUM: prefix_sum.cu scan.o Makefile
	nvcc -o PREFIX_SUM prefix_sum.cu scan.o $(NVCC_OPTS)

scan.o: scan.cu
	nvcc -c scan.cu $(NVCC_OPTS)

clean:
	rm -f  *.o *.sass *.png _cuobjdump* _app* *config *txt* *.log PREFIX_SUM
