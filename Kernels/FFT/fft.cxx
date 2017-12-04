#include <fftw3-mpi.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;






int main(int argc, char **argv)
{
	const ptrdiff_t N0 = 100, N1 = 100;

	fftw_plan plan;
	fftw_complex *data;
	ptrdiff_t alloc_local, local_n0, local_0_start, i, j;

#ifdef _OPENMP
	cout << "Threaded FFTW" << endl;
	auto provided = 0;
	auto required = MPI_THREAD_FUNNELED;
	MPI_Init_thread(&argc, &argv, required, &provided);
	if (provided < required) {
		// We don't have the required MPI support, throw an error
		auto msg = std::stringstream{};
		msg << "Got insufficient MPI support for threads [required=" << required;
		msg << ", provided=" << provided << "]\n";
		throw runtime_error(msg.str());
	}
	// If I have MPI+OpenMP, then FFTW requires at least MPI_THREAD_FUNNELED
	// http://www.fftw.org/fftw3_doc/Combining-MPI-and-Threads.html#Combining-MPI-and-Threads
	auto max_threads = omp_get_max_threads();
	std::cout << "Configuring with max threads: " << max_threads << "\n";
	auto threads_ok = fftw_init_threads();
	if (!threads_ok) {
		throw runtime_error("An error occurred initializing the FFTW OMP library");
	}
#else
	cout << "Non-threaded FFTW" << endl;
	MPI_Init(&argc, &argv);
#endif

	fftw_mpi_init();
#ifdef _OPENMP
	fftw_plan_with_nthreads(max_threads);
#endif

	/* get local data size and allocate */
	alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
			&local_n0, &local_0_start);
	data = fftw_alloc_complex(alloc_local);

	/* create plan for in-place forward DFT */
	plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD,
			FFTW_FORWARD, FFTW_ESTIMATE);    

	/* initialize data to some function my_function(x,y) */
	for (int i = 0; i < local_n0; ++i) {
		for (j = 0; j < N1; ++j) {
			data[i*N1 + j][0] = local_0_start + i + j;
			data[i*N1 + j][1] = local_0_start*i*j;
		}
	}
	/* compute transforms, in-place, as many times as desired */
	for (int ii = 0; ii < 20000; ++ii) {
		fftw_execute(plan);
	}

	fftw_destroy_plan(plan);

	MPI_Finalize();
}

