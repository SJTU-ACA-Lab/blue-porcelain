all:
	nvcc -cudart shared ${CUFILES} -o ${EXECUTABLE} -arch=sm_70
clean:
	rm -f *~ ${EXECUTABLE} *txt* *.log
