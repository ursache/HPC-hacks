FFTW_HOME=/users/fgilles/Projects/Libs/FFT/fftw-3.3.7
CXXFLAGS += -L$(FFTW_HOME)/lib -I$(FFTW_HOME)/include -lfftw3_mpi -lfftw3 -lfftw3_omp
CXXFLAGS += -O3
CXXFLAGS += -std=c++11
CXXFLAGS += -qopenmp
CXXFLAGS += -lmpifort -lmpi 


all:
	mpiicpc $(CXXFLAGS) fft.cxx -o fft
