/**
 * SQUIC CPP example
 *
 * This is an example code for using libSQUIC in CPP. Here we are using Armadillo(http://arma.sourceforge.net) for simplicity ( it is not necessary).
 *
 * Compile Commaned ( e.g., for debuging):
 * g++ -std=c++11 -O0 -g -fsanitize=address   main.cpp - o main.exe -larmadillo -/Location/oF/libSQUIC  -lSQUIC
 *
 * Note you will need to set the path folder of libSQUIC for runtime, for example:
 *
 * export DYLD_LIBRARY_PATH=/Location/Of/libSQUIC (for Mac)
 * export LD_LIBRARY_PATH=/Location/Of/libSQUIC   (for Linux)
 * @author     Aryan Eftekhari
 * @date       2021
 */


// Armadillo libs
#include <armadillo>


// SQUIC Library iterface
extern "C"
{

	/**
	 * @brief      Creat buffer at given location
	 *
	 * @param      buffer  The buffer
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memset_integer(long *&buffer, long length);

	/**
	 * @brief      Creat buffer at given location
	 *
	 * @param      buffer  The buffer
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memset_double(double *&buffer, long length);


	/**
	 * @brief      Creat a new copy of the buffer
	 *
	 * @param      buffer  The copy buffer
	 * @param      values  The buffer to be copied
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memcopy_integer(long *&buffer, long *values, long length);

	/**
	 * @brief      Creat a new copy of the buffer
	 *
	 * @param      buffer  The copy buffer
	 * @param      values  The buffer to be copied
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memcopy_double(double *&buffer, double *values, long length);

	/**
	 * @brief      Free the buffer
	 *
	 * @param      buffer  The buffer
	 */
	void SQUIC_CPP_UTIL_memfree_integer(long *&buffer);

	/**
	 * @brief      Free the buffer
	 *
	 * @param      buffer  The buffer
	 */
	void SQUIC_CPP_UTIL_memfree_double(double *&buffer);

	/**
	 * @brief      SQUIC CPP Interface
	 *
	 * @param[in]        mode            Runtime mode values [0,1,2,3,4] use the "block" algorithem and [5,6,7,8,9] use the "scalar" algorithem (Recomended "0").
	 * @param[in]        p               Number of random variables
	 * @param[in]        n_train1        Number of samples in the training set
	 * @param[in]        Y_train1        Data pointer for training dataset column-major (p x n1)
	 * @param[in]        n_train2        Number of samples in validation-training dataset !!! If n_train2=0 Y2 is ignored !!!
	 * @param[in]        Y_train2        Data pointer for validation-training dataset column-major (p x n_train1) !!! If n2=0 Y2 is ignored !!!
	 * @param[in]        lambda          Scalar sparsity parameter
	 * @param[in]        M_rinx          M matrix row index
	 * @param[in]        M_cptr          M matrix column pointer
	 * @param[in]        M_val           M matrix value
	 * @param[in]        M_nnz           M matrix number of nonzeros
	 * @param[in]        max_iter        Maximum Netwon iterations !!! max_iter=0 will return sample covaraince matrix in ouput iC !!!
	 * @param[in]        drop_tol        Drop tolerance for approximate inversion (drop_tol>0)
	 * @param[in]        term_tol        Termination tolerance (term_tol>0)
	 * @param[in]        verbose         Verbose level
	 * @param[in/out]    X_rinx          Percision matrix row index !!! Intial value X0_rinx is passed in here !!!
	 * @param[in/out]    X_cptr          Percision matrix column pointer !!! Intial value X0_cptr is passed in here !!!
	 * @param[in/out]    X_val           Percision matrix value !!! Intial value X0_val is passed in here !!!
	 * @param[in/out]    X_nnz           Percision matrix nnz  !!! Intial value X_nnz is passed in here !!!
	 * @param[in/out]    W_rinx          Covariance matrix row index !!! Intial value W0_rinx is passed in here !!!
	 * @param[in/out]    W_cptr          Covariance matrix column pointer !!! Intial value W0_rinx is passed in here !!!
	 * @param[in/out]    W_val           Covariance matrix value !!! Intial value W0_rinx is passed in here !!!
	 * @param[in/out]    W_nnz           Covariance matrix nnz  !!! Intial value W0_rinx is passed in here !!!
	 * @param[out]       info_num_iter   Information number of newton iterations performed
	 * @param[out]       info_times      Information 6 element array of times for computing 1)total 2)sample covaraince 3)optimization 4)factorization 5)approximate inversion 6)coordinate update
	 * @param[out]       info_objective  Objective function value !!! this array must be of length max_iter when passed in. Upon ouput only info_num_iter element will be written to !!!
	 * @param[out]       info_dgap       Duality gap
	 * @param[out]       info_logdetX_Y1 Log determinant of X
	 * @param[out]       info_trXS_Y2    Trace(X * S_test), where S_test is sample covaraince fromed from Y_test
	 */
	void SQUIC_CPP(
	    int mode,
	    long p,
	    long n1, double *Y1,
	    long n2, double *Y2,
	    double lambda,
	    long *M_rinx, long *M_cptr, double *M_val, long M_nnz,
	    int max_iter, double drop_tol, double term_tol, int verbose,
	    long *&X_rinx, long *&X_cptr, double *&X_val, long &X_nnz,
	    long *&W_rinx, long *&W_cptr, double *&W_val, long &W_nnz,
	    int &info_num_iter,
	    double *&info_times,	 //length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
	    double *&info_objective, // length must be size max_iter
	    double &info_dgap,
	    double &info_logdetX_Y1,
	    double &info_trXS_Y2);
}

using namespace std;
using namespace arma;

int main(int argc, char *argv[]) {
	// cout << argc << endl;

	if (argc != 5 + 1) {
		printf("Error:[int:p][int:n][int:max_iter][double:drop_tol][double:lambda]\n");
		return 0;
	}

	long p = atoi(argv[1]);
	long n = atoi(argv[2]);
	int max_iter = atoi(argv[3]);
	double drop_tol = atof(argv[4]);
	double lambda = atof(argv[5]);

	arma_rng::set_seed(10);

	///////////////////////////////////////////
	// Generate true inverse covariance matrix
	///////////////////////////////////////////
	printf("# Generate true inverse covariance matrix: Theta=Sigma^{-1}=diag[-.5 1.25 -.5]\n");
	arma::sp_mat Theta_true(p, p), L(p, p);

	L(0, 0) = Theta_true(0, 0) = 1.25;
	L(1, 0) = Theta_true(1, 0) = -.5;
	for (int i = 1; i < p - 1; ++i) {
		Theta_true(i - 1, i) = -.5;
		L(i, i) = Theta_true(i, i) = 1.25;
		L(i + 1, i) = Theta_true(i + 1, i) = -.5;
	}
	Theta_true(p - 2, p - 1) = -.5;
	L(p - 1, p - 1) = Theta_true(p - 1, p - 1) = 1.25;

	///////////////////////////////////////////
	// Generate synthetic data
	///////////////////////////////////////////
	printf("# Computing choleksy factor and generating synthetic data\n");

	// manually compute Cholesky decomposition of a tridiagonal matrix
	for (int i = 0; i < p - 1; ++i) {
		L(i, i) = sqrt(L(i, i));
		L(i + 1, i) /= L(i, i);
		L(i + 1, i + 1) -= L(i + 1, i) * L(i + 1, i);
	}
	L(p - 1, p - 1) = sqrt(L(p - 1, p - 1));

	arma::mat Z = arma::randn<arma::mat>(p, n);
	arma::mat Y(p, n);

	//std::cout << "Z" << std::endl;
	//Z.print();

	// Y=L^{-T}Z
	for (long j = 0; j < n; j++) {
		Y(p - 1, j) = Z(p - 1, j) / L(p - 1, p - 1);
	}

	for (long i = p - 2; i >= 0; --i) {
		for (long j = 0; j < n; j++) {
			Y(i, j) = (Z(i, j) - L(i + 1, i) * Y(i + 1, j)) / L(p - 1, p - 1);
		}
	}

	//std::cout << "Z" << std::endl;
	//Z.print();
	//std::cout << "Y" << std::endl;
	//Y.print();

	///////////////////////////////////////////
	// Calling SQUIC
	///////////////////////////////////////////
	printf("# Calling SQUIC\n");


	int mode = 0;

	// long p = 100; !set from input
	// long n = 50; !set from input

	// double lambda =.5; !set from input
	long *M_rinx;
	long *M_cptr;
	double *M_val;
	long M_nnz = 0;

	// Set an intial guess of identity
	long *X_rinx  = new long[p];
	long *X_cptr  = new long[p + 1];
	double *X_val = new double[p];
	long X_nnz =  p;

	long *W_rinx = new long[p];
	long *W_cptr = new long[p + 1];
	double *W_val = new double[p];
	long W_nnz = p;

	X_cptr[0] = 0;
	W_cptr[0] = 0;
	for (long i = 0; i < p; ++i) {
		X_rinx[i] = i;
		X_cptr[i + 1] = i + 1;
		X_val[i] = 1.0;

		W_rinx[i] = i;
		W_cptr[i + 1] = i + 1;
		W_val[i] = 1.0;
	}

	// int max_iter =10; !set from input
	// double drop_tol = 1e-6;  !set from input
	double term_tol = 1e-6;
	int verbose = 1;

	int info_num_iter = -1;
	double* info_times = new double[6];
	double* info_objective = new double[max_iter];
	double info_dgap = -1;
	double info_logdetx = -1;
	double info_trXS_test = -1;

	long n_test = 0;
	double *Y_test;


	SQUIC_CPP(
	    mode,
	    // Number of random variables
	    p,
	    // Training dataset
	    n, Y.memptr(),
	    // Testing dataset
	    n_test, Y_test,
	    // Regulaization Term
	    lambda, M_rinx, M_cptr, M_val, M_nnz,
	    // Optimization Paramters
	    max_iter, drop_tol, term_tol, verbose,
	    // Intial X0 and W0 are provided, and the end of the routing the final values of X and W are written
	    X_rinx, X_cptr, X_val, X_nnz,
	    W_rinx, W_cptr, W_val, W_nnz,
	    // Run statistics and information
	    info_num_iter,
	    info_times,		// /length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
	    info_objective, // length must be size max_iter
	    info_dgap,
	    info_logdetx,
	    info_trXS_test);



	printf("INFO: obj val:[ ");
	for (int i = 0; i < info_num_iter; ++i) {
		printf("%e " , info_objective[i]);
	}
	printf("]\n");



	printf("INFO: time_total:  %e\n" , info_times[0]);
	printf("INFO: time_impcov: %e\n" , info_times[1]);
	printf("INFO: time_optimz: %e\n" , info_times[2]);
	printf("INFO: time_factor: %e\n" , info_times[3]);
	printf("INFO: time_aprinv: %e\n" , info_times[4]);
	printf("INFO: time_updte:  %e\n" , info_times[5]);

	delete [] info_times;
	delete [] info_objective;




	// Put recovered matricies in the aramadillo
	arma::sp_mat iC(p, p);
	arma::sp_mat C(p, p);


	iC.sync();
	C.sync();

	// Making space for the elements
	iC.mem_resize(X_nnz);
	C.mem_resize(W_nnz);

	// Copying elements
	std::copy(X_rinx, X_rinx + X_nnz, arma::access::rwp(iC.row_indices));
	std::copy(X_cptr, X_cptr + p + 1, arma::access::rwp(iC.col_ptrs));
	std::copy(X_val, X_val + X_nnz, arma::access::rwp(iC.values));
	arma::access::rw(iC.n_rows) = p;
	arma::access::rw(iC.n_cols) = p;
	arma::access::rw(iC.n_nonzero) = X_nnz;

	std::copy(W_rinx, W_rinx + W_nnz, arma::access::rwp(C.row_indices));
	std::copy(W_cptr, W_cptr + p + 1, arma::access::rwp(C.col_ptrs));
	std::copy(W_val, W_val + W_nnz, arma::access::rwp(C.values));
	arma::access::rw(C.n_rows) = p;
	arma::access::rw(C.n_cols) = p;
	arma::access::rw(C.n_nonzero) = W_nnz;


	delete [] X_rinx;
	delete [] X_cptr;
	delete [] X_val;

	delete [] W_rinx;
	delete [] W_cptr;
	delete [] W_val;

	// Make the matrix dense and print ( we make it dense for visulaization !)
	printf("iC:\n");
	arma::mat(iC).print();

	// Make the matrix dense and print ( we make it dense for visulaization !)
	printf("C:\n");
	arma::mat(C).print();


	printf("C x iC:\n");
	arma::mat(C * iC).print();


	return 0;
}
