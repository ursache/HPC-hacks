#
#	CPU_GFLOPS = 8.8
#	GPU_GFLOPS = 300
	GPU_FLAG   = USEGPU

	#DEFINES = -DMPI
	#DEFINES += -DVERBOSE_PRINT
	#DEFINES += -DFERMI_DGEMM_VERBOSE_PRINT
	#DEFINES += -DLIBSCI
	#DEFINES += -DINTERLEAVE
	#DEFINES += -DPINNED


	CC        = gcc 
	CPP       = g++
	NVCC      = nvcc
	FORT      = gfortran

	ARCH      = ar
	ARCHFLAGS = cr
	RANLIB    = ranlib

	OPTS         = -O3 -g -fopenmp 
	NVOPTS       = -O3 -arch=sm_35 
	LDOPTS       =  

	CUDADIR      = /ssoft/nvidia/cuda-7.0/RH6/all/x86_E5v2/
	LIBS		    += -L$(CUDADIR)/lib64/
 
	#LIBS         += -fopenmp -L$(BLASLIBDIR) $(LIBMAGMA) $(LIBMAGMABLAS) -lgfortran -lcublas -lcudart -lcuda -lpthread -lm -mkl_sequential -lmkl_gf_lp64 -lmkl_core 
	LIBS         += -fopenmp -L$(BLASLIBDIR) -lgfortran -lcublas -lcudart -lcuda -lpthread -lm -mkl=sequential 

	HOMEDIR      = /home/fgilles/

	INC          = -I$(CUDADIR)/include/
	
