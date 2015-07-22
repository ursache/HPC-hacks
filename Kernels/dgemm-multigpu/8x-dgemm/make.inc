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


	CC        = icc -vec-report6
	CPP       = g++
	NVCC      = nvcc
	FORT      = gfortran

	ARCH      = ar
	ARCHFLAGS = cr
	RANLIB    = ranlib

	OPTS         = -O3 -g -openmp 
	NVOPTS       = -O3 -arch=sm_35 
	LDOPTS       =  

	CUDADIR      = /cm/shared/apps/cuda60/toolkit/6.0.37/
	LIBS		    += -L$(CUDADIR)/lib64/
 
	#LIBS         += -fopenmp -L$(BLASLIBDIR) $(LIBMAGMA) $(LIBMAGMABLAS) -lgfortran -lcublas -lcudart -lcuda -lpthread -lm -mkl_sequential -lmkl_gf_lp64 -lmkl_core 
	LIBS         += -fopenmp -L$(BLASLIBDIR) -lgfortran -lcublas -lcudart -lcuda -lpthread -lm -mkl=sequential 

	HOMEDIR      = /home/fgilles/

	INC          = -I$(CUDADIR)/include/
	
