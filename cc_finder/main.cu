/*****************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

*****************************************************************/


#include "complex.cuh"
#include "ccf_microlensing.cuh"
#include "ccf_read_write_files.cuh"
#include "parse.hpp"
#include "random_star_field.cuh"
#include "star.cuh"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>



/*constants to be used*/
constexpr int OPTS_SIZE = 2 * 12;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-k", "--kappa_tot",
	"-ks", "--kappa_smooth",
	"-s", "--shear",
	"-t", "--theta_e",
	"-np", "--num_phi",
	"-nb", "--num_branches",
	"-ns", "--num_stars",
	"-rs", "--random_seed",
	"-sf", "--starfile",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};


/*default input option values*/
double kappa_tot = 0.3;
double kappa_smooth = 0.03;
double shear = 0.3;
double theta_e = 1.0;
int num_phi = 50;
int num_branches = 1;
int num_stars = 137;
int random_seed = 0;
std::string starfile = "";
std::string outfile_prefix = "./";
std::string outfile_type = ".bin";

/*default derived parameter values
upper and lower mass cutoffs,
<m>, and <m^2>*/
double m_lower = 1.0;
double m_upper = 1.0;
double mean_mass = 1.0;
double mean_squared_mass = 1.0;



/************************************************************
BEGIN structure definitions and function forward declarations
************************************************************/

/************************************
Print the program usage help message

\param name -- name of the executable
************************************/
void display_usage(char* name);

/****************************************************
function to print out progress bar of loops
examples: [====    ] 50%       [=====  ] 62%

\param icurr -- current position in the loop
\param imax -- maximum position in the loop
\param num_bars -- number of = symbols inside the bar
				   default value: 50
****************************************************/
void print_progress(int icurr, int imax, int num_bars = 50);

/*********************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param name -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
*********************************************************************/
bool CUDAError(const char* name, bool sync, const char* file, const int line);

/**********************************************************
END structure definitions and function forward declarations
**********************************************************/



int main(int argc, char* argv[])
{

	/*set precision for printing numbers to screen*/
	std::cout.precision(7);

	/*if help option has been input, display usage message*/
	if (cmdOptionExists(argv, argv + argc, "-h") || cmdOptionExists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/*if there are input options, but not an even number (since all options
	take a parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name*/
	if ((argc - 1) % 2 != 0)
	{
		std::cerr << "Error. Not enough values for options.\n";
		display_usage(argv[0]);
		return -1;
	}

	/*check that all options given are valid. use step of 2 since all input
	options take parameters (assumed to be given immediately after the option)
	start at 1, since first array element, argv[0], is program name*/
	for (int i = 1; i < argc; i += 2)
	{
		if (!cmdOptionValid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid option input.\n";
			display_usage(argv[0]);
			return -1;
		}
	}


	/******************************************************************************
	BEGIN read in options and values, checking correctness and exiting if necessary
	******************************************************************************/

	char* cmdinput = nullptr;

	for (int i = 1; i < argc; i += 2)
	{
		cmdinput = cmdOptionValue(argv, argv + argc, argv[i]);

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			if (validDouble(cmdinput))
			{
				kappa_tot = std::strtod(cmdinput, nullptr);
			}
			else
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_smooth"))
		{
			if (validDouble(cmdinput))
			{
				kappa_smooth = std::strtod(cmdinput, nullptr);
			}
			else
			{
				std::cerr << "Error. Invalid kappa_smooth input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--shear"))
		{
			if (validDouble(cmdinput))
			{
				shear = std::strtod(cmdinput, nullptr);
			}
			else
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_e"))
		{
			if (validDouble(cmdinput))
			{
				theta_e = std::strtod(cmdinput, nullptr);
				if (theta_e < std::numeric_limits<double>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be > " << std::numeric_limits<double>::min() << "\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid theta_e input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-np") || argv[i] == std::string("--num_phi"))
		{
			if (validDouble(cmdinput))
			{
				num_phi = static_cast<int>(std::strtod(cmdinput, nullptr));
				if (num_phi < 1 || num_phi % 2 != 0)
				{
					std::cerr << "Error. Invalid num_phi input. num_phi must be an even integer > 0\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid num_phi input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-nb") || argv[i] == std::string("--num_branches"))
		{
			if (validDouble(cmdinput))
			{
				num_branches = static_cast<int>(std::strtod(cmdinput, nullptr));
				if (num_branches < 1)
				{
					std::cerr << "Error. Invalid num_branches input. num_branches must be an integer > 0\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid num_branches input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ns") || argv[i] == std::string("--num_stars"))
		{
			if (validDouble(cmdinput))
			{
				num_stars = static_cast<int>(std::strtod(cmdinput, nullptr));
				if (num_stars < 1)
				{
					std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
					return -1;
				}
			}
			else
			{
				std::cerr << "Error. Invalid num_stars input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-rs") || argv[i] == std::string("--random_seed"))
		{
			if (validDouble(cmdinput))
			{
				random_seed = static_cast<int>(std::strtod(cmdinput, nullptr));
			}
			else
			{
				std::cerr << "Error. Invalid random_seed input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			starfile = cmdinput;
		}
		else if (argv[i] == std::string("-ot") || argv[i] == std::string("--outfile_type"))
		{
			outfile_type = cmdinput;
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			outfile_prefix = cmdinput;
		}
	}

	if (num_phi % (2 * num_branches) != 0)
	{
		std::cerr << "Error. Invalid num_phi input. num_phi must be a multiple of 2*num_branches\n";
		return -1;
	}

	/****************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	****************************************************************************/


	/*check that a CUDA capable device is present*/
	cudaSetDevice(0);
	if (CUDAError("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/*if star file is specified, set num_stars, m_lower, m_upper,
	mean_mass, and mean_squared_mass based on star information*/
	if (starfile != "")
	{
		std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

		if (!read_star_params<double>(num_stars, m_lower, m_upper, mean_mass, mean_squared_mass, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n";
	}


	/*variable to hold kappa_star*/
	double kappa_star = kappa_tot - kappa_smooth;

	/*average magnification of the system*/
	double mu_ave = 1.0 / ((1.0 - kappa_tot) * (1.0 - kappa_tot) - shear * shear);

	/*radius needed for number of stars*/
	double rad = std::sqrt(num_stars * mean_mass * theta_e * theta_e / kappa_star);

	std::cout << "Number of stars used: " << num_stars << "\n";

	/*number of roots to be found*/
	int num_roots = 2 * num_stars;


	/**********************
	BEGIN memory allocation
	**********************/

	star<double>* stars = nullptr;
	Complex<double>* ccs_init = nullptr;
	Complex<double>* ccs = nullptr;
	bool* fin = nullptr;
	double* errs = nullptr;
	Complex<double>* caustics = nullptr;
	int* has_nan = nullptr;

	/*allocate memory for stars*/
	cudaMallocManaged(&stars, num_stars * sizeof(star<double>));
	if (CUDAError("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for array of critical curve positions*/
	cudaMallocManaged(&ccs_init, (num_phi + 1) * num_roots * sizeof(Complex<double>));
	if (CUDAError("cudaMallocManaged(*ccs_init)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for array of transposed critical curve positions*/
	cudaMallocManaged(&ccs, (num_phi + 1) * num_roots * sizeof(Complex<double>));
	if (CUDAError("cudaMallocManaged(*ccs)", false, __FILE__, __LINE__)) return -1;

	/*array to hold t/f values of whether or not roots have been found to desired precision*/
	cudaMallocManaged(&fin, num_branches * 2 * num_roots * sizeof(bool));
	if (CUDAError("cudaMallocManaged(*fin)", false, __FILE__, __LINE__)) return -1;

	/*array to hold root errors*/
	cudaMallocManaged(&errs, (num_phi + 1) * num_roots * sizeof(double));
	if (CUDAError("cudaMallocManaged(*errs)", false, __FILE__, __LINE__)) return -1;

	/*array to hold caustic positions*/
	cudaMallocManaged(&caustics, (num_phi + 1) * num_roots * sizeof(Complex<double>));
	if (CUDAError("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return -1;

	/*variable to hold has_nan*/
	cudaMallocManaged(&has_nan, sizeof(int));
	if (CUDAError("cudaMallocManaged(*has_nan)", false, __FILE__, __LINE__)) return -1;

	/********************
	END memory allocation
	********************/


	/**************************
	BEGIN populating star array
	**************************/

	std::cout << "\n";

	if (starfile == "")
	{
		std::cout << "Generating star field...\n";

		/*generate random star field if no star file has been given
		if random seed is provided, use it,
		uses default star mass of 1.0*/
		if (random_seed != 0)
		{
			generate_star_field<double>(stars, num_stars, rad, 1.0, random_seed);
		}
		else
		{
			random_seed = generate_star_field<double>(stars, num_stars, rad, 1.0);
		}

		std::cout << "Done generating star field.\n";
	}
	else
	{
		/*ensure random seed is 0 to denote that stars come from external file*/
		random_seed = 0;

		std::cout << "Reading star field from file " << starfile << "\n";

		/*reading star field from external file*/
		if (!read_star_file<double>(stars, num_stars, starfile))
		{
			std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done reading star field from file " << starfile << "\n";
	}

	/************************
	END populating star array
	************************/


	/*set boolean (int) of errors having nan values to false (0)*/
	*has_nan = 0;

	/*initialize roots for phi=pi to lie at starpos +/- 1*/
	for (int j = 0; j < num_branches; j++)
	{
		for (int i = 0; i < num_stars; i++)
		{
			int index = (num_phi / (2 * num_branches) + j * num_phi / num_branches) * num_roots;
			ccs_init[index + i ] = stars[i].position + 1.0;
			ccs_init[index + i + num_stars] = stars[i].position - 1.0;
		}
	}

	/*initialize values of whether roots have been found to false
	twice the number of roots for a single value of phi, times the
	number of branches, because we will be growing roots for two
	values of phi simultaneously for each branch*/
	for (int i = 0; i < num_branches * 2 * num_roots; i++)
	{
		fin[i] = false;
	}

	for (int i = 0; i < (num_phi + 1) * num_roots; i++)
	{
		errs[i] = 0.0;
	}

	/*number of threads per block, and number of blocks per grid
	uses empirical optimum values for maximum number of threads and blocks*/
	
	int numThreads_x = 128;
	int numThreads_y = num_branches;
	int numBlocks_x = static_cast<int>((2 * num_roots - 1) / numThreads_x) + 1;
	if (numBlocks_x > 32768 || numBlocks_x < 1)
	{
		numBlocks_x = 32768;
	}
	int numBlocks_y = static_cast<int>((num_branches - 1) / numThreads_y) + 1;
	if (numBlocks_y > 32768 || numBlocks_y < 1)
	{
		numBlocks_y = 32768;
	}
	dim3 blocks(numBlocks_x, numBlocks_y);
	dim3 threads(numThreads_x, numThreads_y);


	/*number of iterations to use for root finding
	empirically, 30 seems to be roughly the amount needed*/
	int num_iters = 30;

	/*start and end time for timing purposes*/
	std::chrono::high_resolution_clock::time_point starttime;
	std::chrono::high_resolution_clock::time_point endtime;

	/*begin finding initial roots*/
	std::cout << "\nFinding initial roots...\n";

	/*get current time at start of loop*/
	starttime = std::chrono::high_resolution_clock::now();

	/*each iteration of this loop calculates updated positions
	of all numroots roots in parallel
	ideally, the number of loop iterations is enough to ensure that
	all roots are found to the desired accuracy*/
	for (int i = 0; i < num_iters; i++)
	{
		/*display percentage done
		uses default of printing progress bar of length 50*/
		print_progress(i, num_iters - 1);

		/*start kernel and perform error checking*/
		find_critical_curve_roots_kernel<double> << < blocks, threads >> > (stars, num_stars, kappa_smooth, shear, theta_e, ccs_init, num_roots, 0, num_phi, num_branches, fin);
		if (CUDAError("find_critical_curve_roots_kernel", true, __FILE__, __LINE__)) return -1;
	}

	/*get current time at end of loop, and calculate duration in milliseconds*/
	endtime = std::chrono::high_resolution_clock::now();
	double t_init_roots = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;

	std::cout << "\nDone finding roots. Elapsed time: " << t_init_roots << " seconds.\n";


	/*calculate errors in 1/mu for initial roots*/
	find_errors_kernel<double> << < blocks, threads >> > (ccs_init, num_roots, stars, num_stars, kappa_smooth, shear, theta_e, 0, num_phi, num_branches, errs);
	if (CUDAError("find_errors_kernel", false, __FILE__, __LINE__)) return -1;

	has_nan_err_kernel<double> << < blocks, threads >> > (errs, (num_phi + 1) * num_roots, has_nan);
	if (CUDAError("has_nan_err_kernel", true, __FILE__, __LINE__)) return -1;

	if (*has_nan)
	{
		std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
		return -1;
	}

	/*find max error and print*/
	int num_errs = (num_phi + 1) * num_roots;
	while (num_errs > 1)
	{
		if (num_errs % 2 != 0)
		{
			errs[num_errs - 2] = std::fmax(errs[num_errs - 2], errs[num_errs - 1]);
			num_errs -= 1;
		}
		num_errs /= 2;
		max_err_kernel<double> << < blocks, threads >> > (errs, num_errs);
		if (CUDAError("max_err_kernel", true, __FILE__, __LINE__)) return -1;
	}
	double max_error = errs[0];
	std::cout << "Maximum error in 1/mu: " << max_error << "\n";


	/*reduce number of iterations needed, as roots should stay close to previous positions*/
	num_iters = 25;

	/*begin finding critical curves*/
	std::cout << "\nFinding critical curve positions...\n";

	starttime = std::chrono::high_resolution_clock::now();

	/*the outer loop will step through different values of phi
	we use num_phi/2 steps, as we will be working our way out
	from the middle of the array of roots towards the two endpoints
	simultaneously (i.e., from phi=pi to phi=0 and phi=2pi simultaneously)*/
	for (int i = 1; i <= num_phi / (2 * num_branches); i++)
	{

		/*set critical curve array elements to be equal to last roots
		reuse fin array for each set of phi roots*/
		prepare_roots_kernel<double> << < blocks, threads >> > (ccs_init, num_roots, i, num_phi, num_branches, fin);
		if (CUDAError("prepare_roots_kernel", false, __FILE__, __LINE__)) return -1;

		/*solve roots for current values of phi = pi +/- i*2pi/num_phi*/
		for (int j = 0; j < num_iters; j++)
		{
			find_critical_curve_roots_kernel<double> << < blocks, threads >> > (stars, num_stars, kappa_smooth, shear, theta_e, ccs_init, num_roots, i, num_phi, num_branches, fin);
			if (CUDAError("find_critical_curve_roots_kernel", false, __FILE__, __LINE__)) return -1;
		}
		/*only perform synchronization call after roots have all been found
		this allows the print_progress call in the outer loop to accurately display
		the amount of work done so far
		one could move the synchronization call outside of the outer loop for a
		slight speed-up, at the cost of not knowing how far along in the process
		the computations have gone*/
		if (i * 100 / (num_phi / (2 * num_branches)) > (i - 1) * 100 / (num_phi / (2 * num_branches)))
		{
			cudaDeviceSynchronize();
			if (CUDAError("cudaDeviceSynchronize", false, __FILE__, __LINE__)) return -1;
			print_progress(i, num_phi / (2 * num_branches));
		}
	}

	endtime = std::chrono::high_resolution_clock::now();
	double t_ccs = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "\nDone finding critical curve positions. Elapsed time: " << t_ccs << " seconds.\n";


	/*find max error in 1/mu over whole critical curve array and print*/
	std::cout << "\nFinding maximum error in 1/mu over all calculated critical curve positions...\n";

	*has_nan = 0;

	for (int i = 0; i <= num_phi / (2 * num_branches); i++)
	{
		find_errors_kernel<double> << < blocks, threads >> > (ccs_init, num_roots, stars, num_stars, kappa_smooth, shear, theta_e, i, num_phi, num_branches, errs);
		if (CUDAError("find_errors_kernel", false, __FILE__, __LINE__)) return -1;
	}

	has_nan_err_kernel<double> << < blocks, threads >> > (errs, (num_phi + 1) * num_roots, has_nan);
	if (CUDAError("has_nan_err_kernel", true, __FILE__, __LINE__)) return -1;

	if (*has_nan)
	{
		std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
		return -1;
	}

	num_errs = (num_phi + 1) * num_roots;
	while (num_errs > 1)
	{
		if (num_errs % 2 != 0)
		{
			errs[num_errs - 2] = std::fmax(errs[num_errs - 2], errs[num_errs - 1]);
			num_errs -= 1;
		}
		num_errs /= 2;
		max_err_kernel<double> << < blocks, threads >> > (errs, num_errs);
		if (CUDAError("max_err_kernel", true, __FILE__, __LINE__)) return -1;
	}
	max_error = errs[0];
	std::cout << "Maximum error in 1/mu: " << max_error << "\n";


	std::cout << "\nTransposing critical curve array...\n";
	starttime = std::chrono::high_resolution_clock::now();
	transpose_array_kernel<double> << < blocks, threads >> > (ccs_init, (num_phi + 1), num_roots, ccs);
	if (CUDAError("transpose_array_kernel", true, __FILE__, __LINE__)) return -1;
	endtime = std::chrono::high_resolution_clock::now();
	std::cout << "Done transposing critical curve array. Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0 << " seconds.\n";

	std::cout << "\nFinding caustic positions...\n";
	starttime = std::chrono::high_resolution_clock::now();
	find_caustics_kernel<double> << < blocks, threads >> > (ccs, (num_phi + 1) * num_roots, stars, num_stars, kappa_smooth, shear, theta_e, caustics);
	if (CUDAError("find_caustics_kernel", true, __FILE__, __LINE__)) return -1;
	endtime = std::chrono::high_resolution_clock::now();
	double t_caustics = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "Done finding caustic positions. Elapsed time: " << t_caustics << " seconds.\n";



	/*stream for writing output files
	set precision to 9 digits*/
	std::ofstream outfile;
	outfile.precision(9);

	std::cout << "\nWriting parameter info...\n";
	outfile.open(outfile_prefix + "ccf_parameter_info.txt");
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << (outfile_prefix + "ccf_parameter_info.txt") << "\n";
		return -1;
	}
	outfile << "kappa_tot " << kappa_tot << "\n";
	outfile << "kappa_star " << kappa_star << "\n";
	outfile << "kappa_smooth " << kappa_smooth << "\n";
	outfile << "shear " << shear << "\n";
	outfile << "theta_e " << theta_e << "\n";
	outfile << "mu_ave " << mu_ave << "\n";
	outfile << "lower_mass_cutoff " << m_lower << "\n";
	outfile << "upper_mass_cutoff " << m_upper << "\n";
	outfile << "mean_mass " << mean_mass << "\n";
	outfile << "mean_squared_mass " << mean_squared_mass << "\n";
	outfile << "num_stars " << num_stars << "\n";
	outfile << "rad " << rad << "\n";
	outfile << "random_seed " << random_seed << "\n";
	outfile << "num_roots " << num_roots << "\n";
	outfile << "num_phi " << num_phi << "\n";
	outfile << "num_branches " << num_branches << "\n";
	outfile << "max_error_1/mu " << max_error << "\n";
	outfile << "t_init_roots " << t_init_roots << "\n";
	outfile << "t_ccs " << t_ccs << "\n";
	outfile << "t_caustics " << t_caustics << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << outfile_prefix << "ccf_parameter_info.txt\n";

	std::cout << "\nWriting star info...\n";
	if (!write_star_file<double>(stars, num_stars, outfile_prefix + "ccf_star_info" + outfile_type))
	{
		std::cerr << "Error. Unable to write star info to file " << outfile_prefix << "ccf_star_info" + outfile_type << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << outfile_prefix << "ccf_star_info" + outfile_type << "\n";


	/*write critical curve positions*/
	std::cout << "\nWriting critical curve positions...\n";
	if (outfile_type == ".txt")
	{
		if (!write_re_array<double>(ccs, num_roots, num_phi + 1, outfile_prefix + "ccf_ccs_pos_x" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs x info to file " << outfile_prefix << "ccf_ccs_pos_x" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve x positions to file " << outfile_prefix << "ccf_ccs_pos_x" + outfile_type << "\n";
		if (!write_im_array<double>(ccs, num_roots, num_phi + 1, outfile_prefix + "ccf_ccs_pos_y" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs y info to file " << outfile_prefix << "ccf_ccs_pos_y" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve y positions to file " << outfile_prefix << "ccf_ccs_pos_y" + outfile_type << "\n";
	}
	else
	{
		if (!write_complex_array<double>(ccs, num_roots, num_phi + 1, outfile_prefix + "ccf_ccs_pos" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs info to file " << outfile_prefix << "ccf_ccs_pos" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve positions to file " << outfile_prefix << "ccf_ccs_pos" + outfile_type << "\n";
	}


	/*write caustic positions*/
	std::cout << "\nWriting caustic positions...\n";
	if (outfile_type == ".txt")
	{
		if (!write_re_array<double>(caustics, num_roots, num_phi + 1, outfile_prefix + "ccf_caustics_pos_x" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic x info to file " << outfile_prefix << "ccf_caustics_pos_x" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic x positions to file " << outfile_prefix << "ccf_caustics_pos_x" + outfile_type << "\n";
		if (!write_im_array<double>(caustics, num_roots, num_phi + 1, outfile_prefix + "ccf_caustics_pos_y" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic y info to file " << outfile_prefix << "ccf_caustics_pos_y" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic y positions to file " << outfile_prefix << "ccf_caustics_pos_y" + outfile_type << "\n";
	}
	else
	{
		if (!write_complex_array<double>(caustics, num_roots, num_phi + 1, outfile_prefix + "ccf_caustics_pos" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic info to file " << outfile_prefix << "ccf_caustics_pos" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic positions to file " << outfile_prefix << "ccf_caustics_pos" + outfile_type << "\n";
	}

	std::cout << "\nDone.\n";

	cudaDeviceReset();
	if (CUDAError("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}



void display_usage(char* name)
{
	if (name)
	{
		std::cout << "Usage: " << name << " opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	else
	{
		std::cout << "Usage: programname opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	std::cout << "Options:\n"
		<< "   -h,--help             Show this help message\n"
		<< "   -k,--kappa_tot        Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "   -ks,--kappa_smooth    Specify the smooth convergence. Default value: " << kappa_smooth << "\n"
		<< "   -s,--shear            Specify the shear. Default value: " << shear << "\n"
		<< "   -t,--theta_e          Specify the size of the Einstein radius of a unit mass\n"
		<< "                         star in arbitrary units. Default value: " << theta_e << "\n"
		<< "   -np,--num_phi         Specify the number of steps used to vary phi in the\n"
		<< "                         range [0, 2*pi]. Default value: " << num_phi << "\n"
		<< "   -nb,--num_branches    Specify the number of branches to use in the range\n"
		<< "                         [0, 2*pi]. Default value: " << num_branches << "\n"
		<< "   -ns,--num_stars       Specify the number of stars desired.\n"
		<< "                         Default value: " << num_stars << "\n"
		<< "                         All stars are taken to be of unit mass. If a range of\n"
		<< "                         masses are desired, please input them through a file\n"
		<< "                         as described in the -sf option.\n"
		<< "   -rs,--random_seed     Specify the random seed for the star field generation.\n"
		<< "                         A value of 0 is reserved for star input files.\n"
		<< "                         Default value: " << random_seed << "\n"
		<< "   -sf,--starfile        Specify the location of a star positions and masses\n"
		<< "                         file. Default value: " << starfile << "\n"
		<< "                         The file may be either a whitespace delimited text\n"
		<< "                         file containing valid double precision values for a\n"
		<< "                         star's x coordinate, y coordinate, and mass, in that\n"
		<< "                         order, on each line, or a binary file of star\n"
		<< "                         structures (as defined in this source code). If\n"
		<< "                         specified, the number of stars is determined through\n"
		<< "                         this file and the -ns option is ignored.\n"
		<< "   -ot,--outfile_type    Specify the type of file to be output. Valid options\n"
		<< "                         are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "   -o,--outfile_prefix   Specify the prefix to be used in output filenames.\n"
		<< "                         Default value: " << outfile_prefix << "\n"
		<< "                         Lines of .txt output files are whitespace delimited.\n"
		<< "                         Filenames are:\n"
		<< "                            ccf_parameter_info   various parameter values used\n"
		<< "                                                     in calculations\n"
		<< "                            ccf_star_info        each line contains a star's x\n"
		<< "                                                     coordinate, y coordinate,\n"
		<< "                                                     and mass\n"
		<< "                            ccf_ccs_pos_x        each of the 2*num_stars lines\n"
		<< "                                                     contains num_phi+1 values\n"
		<< "                                                     tracing a root for phi in\n"
		<< "                                                     [0, 2*pi]\n"
		<< "                            ccf_ccs_pos_y\n"
		<< "                            ccf_caustics_pos_x\n"
		<< "                            ccf_caustics_pos_y\n";
}

void print_progress(int icurr, int imax, int num_bars)
{
	std::cout << "\r[";
	for (int i = 0; i < num_bars; i++)
	{
		if (i <= icurr * num_bars / imax)
		{
			std::cout << "=";
		}
		else
		{
			std::cout << " ";
		}
	}
	std::cout << "] " << icurr * 100 / imax << " %";
}

bool CUDAError(const char* name, bool sync, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	/*if the last error message is not a success, print the error code and msg
	and return true (i.e., an error occurred)*/
	if (err != cudaSuccess)
	{
		const char* errMsg = cudaGetErrorString(err);
		std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
		std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
		return true;
	}
	/*if a device synchronization is also to be done*/
	if (sync)
	{
		/*perform the same error checking as initially*/
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			const char* errMsg = cudaGetErrorString(err);
			std::cerr << "CUDA error check for cudaDeviceSynchronize failed at " << file << ":" << line << "\n";
			std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
			return true;
		}
	}
	return false;
}

