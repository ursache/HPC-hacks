#
	CC        = icc
	CPP       = g++
	NVCC      = nvcc
	FORT      = gfortran

	ARCH      = ar
	ARCHFLAGS = cr
	RANLIB    = ranlib

#	OPTS      = -W -Wall -O3 -fomit-frame-pointer -fopenmp  
	OPTS      = -W -Wall -O3 -fopenmp  
	NVOPTS    = --compiler-options -fno-strict-aliasing  -arch sm_35 \
                    -DUNIX -O3
	LDOPTS    = -fopenmp 
 
	#LIB       = -lpthread -lm  
	#LIB      += -lcublas -lcudart -lsci_acc_gnu_47_nv35

	CUDADIR   = /usr/local/cuda-6.0/ 
	#HOMEDIR   = /Users/foureste/

	LIBDIR   += -L$(CUDADIR)/lib64/ -lcublas -lcudart
	INC       = -I$(CUDADIR)/include 

