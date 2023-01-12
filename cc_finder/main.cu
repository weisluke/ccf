/*****************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

*****************************************************************/


#include "complex.cuh"
#include "ccf_microlensing.cuh"
#include "ccf_read_write_files.cuh"
#include "star.cuh"
#include "util.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>


using dtype = double;

/*constants to be used*/
const dtype PI = static_cast<dtype>(3.1415926535898);
constexpr int OPTS_SIZE = 2 * 15;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-k", "--kappa_tot",
	"-s", "--shear",
	"-t", "--theta_e",
	"-ks", "--kappa_star",
	"-r", "--rectangular",
	"-a", "--approx",
	"-ts", "--taylor",
	"-ns", "--num_stars",
	"-sf", "--starfile",
	"-np", "--num_phi",
	"-nb", "--num_branches",
	"-rs", "--random_seed",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};


/*default input option values*/
dtype kappa_tot = static_cast<dtype>(0.3);
dtype shear = static_cast<dtype>(0.3);
dtype theta_e = static_cast<dtype>(1);
dtype kappa_star = static_cast<dtype>(0.27);
int rectangular = 1;
int approx = 1;
int taylor = 1;
int num_stars = 137;
std::string starfile = "";
int num_phi = 50;
int num_branches = 1;
int random_seed = 0;
std::string outfile_type = ".bin";
std::string outfile_prefix = "./";

/*default derived parameter values
upper and lower mass cutoffs,
<m>, and <m^2>*/
dtype m_lower = static_cast<dtype>(1);
dtype m_upper = static_cast<dtype>(1);
dtype mean_mass = static_cast<dtype>(1);
dtype mean_squared_mass = static_cast<dtype>(1);



/************************************
Print the program usage help message

\param name -- name of the executable
************************************/
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
	std::cout 
		<< "Options:\n"
		<< "   -h,--help             Show this help message\n"
		<< "   -k,--kappa_tot        Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "   -s,--shear            Specify the shear. Default value: " << shear << "\n"
		<< "   -t,--theta_e          Specify the size of the Einstein radius of a unit mass\n"
		<< "                         point lens in arbitrary units. Default value: " << theta_e << "\n"
		<< "   -ks,--kappa_star      Specify the convergence in point mass lenses.\n"
		<< "                         Default value: " << kappa_star << "\n"
		<< "   -r,--rectangular      Specify whether the star field should be\n"
		<< "                         rectangular (1) or circular (0). Default value: " << rectangular << "\n"
		<< "   -a,--approx           Specify whether terms for alpha_smooth should be\n"
		<< "                         approximated (1) or exact (0). Default value: " << approx << "\n"
		<< "   -ts,--taylor          Specify the highest degree for the Taylor series of\n"
		<< "                         alpha_smooth. Default value: " << taylor << "\n"
		<< "   -ns,--num_stars       Specify the number of stars desired.\n"
		<< "                         Default value: " << num_stars << "\n"
		<< "                         All stars are taken to be of unit mass. If a range of\n"
		<< "                         masses are desired, please input them through a file\n"
		<< "                         as described in the -sf option.\n"
		<< "   -sf,--starfile        Specify the location of a star positions and masses\n"
		<< "                         file. The file may be either a whitespace delimited\n"
		<< "                         text file containing valid values for a star's x\n"
		<< "                         coordinate, y coordinate, and mass, in that order, on\n"
		<< "                         each line, or a binary file of star structures (as\n"
		<< "                         defined in this source code). If specified, the number\n"
		<< "                         of stars is determined through this file and the -ns\n"
		<< "                         option is ignored.\n"
		<< "   -np,--num_phi         Specify the number of steps used to vary phi in the\n"
		<< "                         range [0, 2*pi]. Default value: " << num_phi << "\n"
		<< "   -nb,--num_branches    Specify the number of branches to use for phi in the\n"
		<< "                         range [0, 2*pi]. Default value: " << num_branches << "\n"
		<< "   -rs,--random_seed     Specify the random seed for star field generation.\n"
		<< "                         A value of 0 is reserved for star input files.\n"
		<< "   -ot,--outfile_type    Specify the type of file to be output. Valid options\n"
		<< "                         are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "   -o,--outfile_prefix   Specify the prefix to be used in output filenames.\n"
		<< "                         Default value: " << outfile_prefix << "\n"
		<< "                         Lines of .txt output files are whitespace delimited.\n"
		<< "                         Filenames are:\n"
		<< "                            ccf_parameter_info   various parameter values used\n"
		<< "                                                     in calculations\n"
		<< "                            ccf_stars            the first item is num_stars\n"
		<< "                                                     followed by binary\n"
		<< "                                                     representations of the\n"
		<< "                                                     star structures\n"
		<< "                            ccf_ccs              the first item is num_roots\n"
		<< "                                                     and the second item is\n"
		<< "                                                     num_phi / num_branches + 1\n"
		<< "                                                     followed by binary\n"
		<< "                                                     representations of the\n"
		<< "                                                     complex critical curve\n"
		<< "                                                     values\n"
		<< "                            ccf_caustics         the first item is num_roots\n"
		<< "                                                     and the second item is\n"
		<< "                                                     num_phi / num_branches + 1\n"
		<< "                                                     followed by binary\n"
		<< "                                                     representations of the\n"
		<< "                                                     complex caustic curve\n"
		<< "                                                     values\n";
}

/*********************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param name -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
*********************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line)
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



int main(int argc, char* argv[])
{

	/*set precision for printing numbers to screen*/
	std::cout.precision(7);

	/*if help option has been input, display usage message*/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
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
		if (!cmd_option_valid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid input syntax. Unknown option " << argv[i] << "\n";
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
		cmdinput = cmd_option_value(argv, argv + argc, argv[i]);

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				kappa_tot = static_cast<dtype>(std::stod(cmdinput));
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--shear"))
		{
			try
			{
				shear = static_cast<dtype>(std::stod(cmdinput));
			}
			catch (...)
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_e"))
		{
			try
			{
				theta_e = static_cast<dtype>(std::stod(cmdinput));
				if (theta_e < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid theta_e input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_star"))
		{
			try
			{
				kappa_star = static_cast<dtype>(std::stod(cmdinput));
				if (kappa_star < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid kappa_star input. kappa_star must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_star input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-r") || argv[i] == std::string("--rectangular"))
		{
			try
			{
				rectangular = std::stoi(cmdinput);
				if (rectangular != 0 && rectangular != 1)
				{
					std::cerr << "Error. Invalid rectangular input. rectangular must be 1 (rectangular) or 0 (circular).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid rectangular input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-a") || argv[i] == std::string("--approx"))
		{
			try
			{
				approx = std::stoi(cmdinput);
				if (approx != 0 && approx != 1)
				{
					std::cerr << "Error. Invalid approx input. approx must be 1 (approximate) or 0 (exact).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid approx input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ts") || argv[i] == std::string("--taylor"))
		{
			try
			{
				taylor = std::stoi(cmdinput);
				if (taylor < 1)
				{
					std::cerr << "Error. Invalid taylor input. taylor must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid taylor input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ns") || argv[i] == std::string("--num_stars"))
		{
			try
			{
				num_stars = std::stoi(cmdinput);
				if (num_stars < 1)
				{
					std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_stars input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			starfile = cmdinput;
		}
		else if (argv[i] == std::string("-np") || argv[i] == std::string("--num_phi"))
		{
			try
			{
				num_phi = std::stoi(cmdinput);
				if (num_phi < 1 || num_phi % 2 != 0)
				{
					std::cerr << "Error. Invalid num_phi input. num_phi must be an even integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_phi input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-nb") || argv[i] == std::string("--num_branches"))
		{
			try
			{
				num_branches = std::stoi(cmdinput);
				if (num_branches < 1)
				{
					std::cerr << "Error. Invalid num_branches input. num_branches must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_branches input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-rs") || argv[i] == std::string("--random_seed"))
		{
			try
			{
				random_seed = std::stoi(cmdinput);
				if (random_seed == 0 && !(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--star_file")))
				{
					std::cerr << "Error. Invalid random_seed input. Seed of 0 is reserved for star input files.\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid random_seed input.\n";
				return -1;
			}
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
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/*if star file is specified, set num_stars, m_lower, m_upper,
	mean_mass, and mean_squared_mass based on star information*/
	if (starfile != "")
	{
		std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

		if (!read_star_params<dtype>(num_stars, m_lower, m_upper, mean_mass, mean_squared_mass, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n";
	}

	/*average magnification of the system*/
	dtype mu_ave = 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear);

	std::cout << "Number of stars used: " << num_stars << "\n";

	Complex<dtype> c = std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
		* Complex<dtype>(
			std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
			std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
			);
	dtype rad = std::sqrt(theta_e * theta_e * num_stars * mean_mass / kappa_star);

	/*number of roots to be found*/
	int num_roots = 2 * num_stars;
	if (rectangular && approx)
	{
		num_roots += static_cast<int>(taylor / 2) * 2;
	}


	/**********************
	BEGIN memory allocation
	**********************/

	star<dtype>* stars = nullptr;
	Complex<dtype>* ccs_init = nullptr;
	Complex<dtype>* ccs = nullptr;
	bool* fin = nullptr;
	dtype* errs = nullptr;
	Complex<dtype>* caustics = nullptr;
	int* has_nan = nullptr;

	/*allocate memory for stars*/
	cudaMallocManaged(&stars, num_stars * sizeof(star<dtype>));
	if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for array of critical curve positions*/
	cudaMallocManaged(&ccs_init, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*ccs_init)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for array of transposed critical curve positions*/
	cudaMallocManaged(&ccs, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*ccs)", false, __FILE__, __LINE__)) return -1;

	/*array to hold t/f values of whether or not roots have been found to desired precision*/
	cudaMallocManaged(&fin, num_branches * 2 * num_roots * sizeof(bool));
	if (cuda_error("cudaMallocManaged(*fin)", false, __FILE__, __LINE__)) return -1;

	/*array to hold root errors*/
	cudaMallocManaged(&errs, (num_phi + num_branches) * num_roots * sizeof(dtype));
	if (cuda_error("cudaMallocManaged(*errs)", false, __FILE__, __LINE__)) return -1;

	/*array to hold caustic positions*/
	cudaMallocManaged(&caustics, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return -1;

	/*variable to hold has_nan*/
	cudaMallocManaged(&has_nan, sizeof(int));
	if (cuda_error("cudaMallocManaged(*has_nan)", false, __FILE__, __LINE__)) return -1;

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
			if (rectangular)
			{
				generate_rectangular_star_field<dtype>(stars, num_stars, c, static_cast<dtype>(1), random_seed);
			}
			else
			{
				generate_circular_star_field<dtype>(stars, num_stars, rad, static_cast<dtype>(1), random_seed);
			}
		}
		else
		{
			if (rectangular)
			{
				random_seed = generate_rectangular_star_field<dtype>(stars, num_stars, c, static_cast<dtype>(1));
			}
			else
			{
				random_seed = generate_circular_star_field<dtype>(stars, num_stars, rad, static_cast<dtype>(1));
			}
		}

		std::cout << "Done generating star field.\n";
	}
	else
	{
		/*ensure random seed is 0 to denote that stars come from external file*/
		random_seed = 0;

		std::cout << "Reading star field from file " << starfile << "\n";

		/*reading star field from external file*/
		if (!read_star_file<dtype>(stars, num_stars, starfile))
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

	/*initialize roots for centers of all branches to lie at starpos +/- 1*/
	for (int j = 0; j < num_branches; j++)
	{
		int center = (num_phi / (2 * num_branches) + j * num_phi / num_branches + j) * num_roots;
		for (int i = 0; i < num_stars; i++)
		{
			ccs_init[center + i ] = stars[i].position + 1;
			ccs_init[center + i + num_stars] = stars[i].position - 1;
		}
		if (rectangular && approx)
		{
			int nroots_extra = static_cast<int>(taylor / 2) * 2;
			for (int i = 0; i < nroots_extra; i++)
			{
				ccs_init[center + 2 * num_stars + i] = c.abs() * Complex<dtype>(std::cos(2 * PI / nroots_extra * i), std::sin(2 * PI / nroots_extra * i));
			}
		}
	}

	/*initialize values of whether roots have been found to false
	twice the number of roots for a single value of phi for each branch,
	times the number of branches, because we will be growing roots for two
	values of phi simultaneously for each branch*/
	for (int i = 0; i < num_branches * 2 * num_roots; i++)
	{
		fin[i] = false;
	}

	for (int i = 0; i < (num_phi + num_branches) * num_roots; i++)
	{
		errs[i] = static_cast<dtype>(0);
	}

	/*number of threads per block, and number of blocks per grid
	uses empirical optimum values for maximum number of threads and blocks*/
	
	int num_threads_z = 1;
	int num_threads_y = 1;
	int num_threads_x = 32;

	int num_blocks_z = num_branches;
	int num_blocks_y = 2;
	int num_blocks_x = static_cast<int>((num_roots - 1) / num_threads_x) + 1;
	
	dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	dim3 threads(num_threads_x, num_threads_y, num_threads_z);


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
	of all roots for the center of each branch in parallel
	ideally, the number of loop iterations is enough to ensure that
	all roots are found to the desired accuracy*/
	for (int i = 0; i < num_iters; i++)
	{
		/*display percentage done
		uses default of printing progress bar of length 50*/
		print_progress(i, num_iters - 1);

		/*start kernel and perform error checking*/
		if (rectangular)
		{
			if (approx)
			{
				find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, taylor, ccs_init, num_roots, 0, num_phi, num_branches, fin);
			}
			else
			{
				find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, ccs_init, num_roots, 0, num_phi, num_branches, fin);
			}
		}
		else
		{
			find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, ccs_init, num_roots, 0, num_phi, num_branches, fin);
		}
		if (cuda_error("find_critical_curve_roots_kernel", true, __FILE__, __LINE__)) return -1;
	}

	/*get current time at end of loop, and calculate duration in milliseconds*/
	endtime = std::chrono::high_resolution_clock::now();
	double t_init_roots = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;

	std::cout << "\nDone finding roots. Elapsed time: " << t_init_roots << " seconds.\n";


	/*calculate errors in 1/mu for initial roots*/
	if (rectangular)
	{
		if (approx)
		{
			find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, taylor, 0, num_phi, num_branches, errs);
		}
		else
		{
			find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, 0, num_phi, num_branches, errs);
		}
	}
	else
	{
		find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 0, num_phi, num_branches, errs);
	}
	if (cuda_error("find_errors_kernel", false, __FILE__, __LINE__)) return -1;

	has_nan_err_kernel<dtype> <<<blocks, threads>>> (errs, (num_phi + num_branches) * num_roots, has_nan);
	if (cuda_error("has_nan_err_kernel", true, __FILE__, __LINE__)) return -1;

	if (*has_nan)
	{
		std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
		return -1;
	}

	/*find max error and print*/
	int num_errs = (num_phi + num_branches) * num_roots;
	while (num_errs > 1)
	{
		if (num_errs % 2 != 0)
		{
			errs[num_errs - 2] = std::fmax(errs[num_errs - 2], errs[num_errs - 1]);
			num_errs -= 1;
		}
		num_errs /= 2;
		max_err_kernel<dtype> <<<blocks, threads>>> (errs, num_errs);
		if (cuda_error("max_err_kernel", true, __FILE__, __LINE__)) return -1;
	}
	dtype max_error = errs[0];
	std::cout << "Maximum error in 1/mu: " << max_error << "\n";


	/*reduce number of iterations needed, as roots should stay close to previous positions*/
	num_iters = 20;

	/*begin finding critical curves*/
	std::cout << "\nFinding critical curve positions...\n";

	starttime = std::chrono::high_resolution_clock::now();

	/*the outer loop will step through different values of phi
	we use num_phi/(2*num_branches) steps, as we will be working our way out
	from the middle of each branch for the array of roots simultaneously*/
	for (int j = 1; j <= num_phi / (2 * num_branches); j++)
	{

		/*set critical curve array elements to be equal to last roots
		fin array is reused each time*/
		prepare_roots_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, j, num_phi, num_branches, fin);
		if (cuda_error("prepare_roots_kernel", false, __FILE__, __LINE__)) return -1;

		/*solve roots for current values of j*/
		for (int i = 0; i < num_iters; i++)
		{
			if (rectangular)
			{
				if (approx)
				{
					find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, taylor, ccs_init, num_roots, j, num_phi, num_branches, fin);
				}
				else
				{
					find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, ccs_init, num_roots, j, num_phi, num_branches, fin);
				}
			}
			else
			{
				find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, ccs_init, num_roots, j, num_phi, num_branches, fin);
			}
			if (cuda_error("find_critical_curve_roots_kernel", false, __FILE__, __LINE__)) return -1;
		}
		/*only perform synchronization call after roots have all been found
		this allows the print_progress call in the outer loop to accurately display
		the amount of work done so far
		one could move the synchronization call outside of the outer loop for a
		slight speed-up, at the cost of not knowing how far along in the process
		the computations have gone*/
		if (j * 100 / (num_phi / (2 * num_branches)) > (j - 1) * 100 / (num_phi / (2 * num_branches)))
		{
			cudaDeviceSynchronize();
			if (cuda_error("cudaDeviceSynchronize", false, __FILE__, __LINE__)) return -1;
			print_progress(j, num_phi / (2 * num_branches));
		}
	}

	endtime = std::chrono::high_resolution_clock::now();
	double t_ccs = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "\nDone finding critical curve positions. Elapsed time: " << t_ccs << " seconds.\n";


	/*find max error in 1/mu over whole critical curve array and print*/
	std::cout << "\nFinding maximum error in 1/mu over all calculated critical curve positions...\n";

	for (int j = 0; j <= num_phi / (2 * num_branches); j++)
	{
		if (rectangular)
		{
			if (approx)
			{
				find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, taylor, j, num_phi, num_branches, errs);
			}
			else
			{
				find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, j, num_phi, num_branches, errs);
			}
		}
		else
		{
			find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, j, num_phi, num_branches, errs);
		}
		if (cuda_error("find_errors_kernel", false, __FILE__, __LINE__)) return -1;
	}

	has_nan_err_kernel<dtype> <<<blocks, threads>>> (errs, (num_phi + num_branches) * num_roots, has_nan);
	if (cuda_error("has_nan_err_kernel", true, __FILE__, __LINE__)) return -1;

	if (*has_nan)
	{
		std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
		return -1;
	}

	num_errs = (num_phi + num_branches) * num_roots;
	while (num_errs > 1)
	{
		if (num_errs % 2 != 0)
		{
			errs[num_errs - 2] = std::fmax(errs[num_errs - 2], errs[num_errs - 1]);
			num_errs -= 1;
		}
		num_errs /= 2;
		max_err_kernel<dtype> <<<blocks, threads>>> (errs, num_errs);
		if (cuda_error("max_err_kernel", true, __FILE__, __LINE__)) return -1;
	}
	max_error = errs[0];
	std::cout << "Maximum error in 1/mu: " << max_error << "\n";


	/*redefine thread and block size to maximize parallelization*/
	num_threads_z = 1;
	num_threads_y = 1;
	num_threads_x = 512;

	num_blocks_z = 1;
	num_blocks_y = 1;
	num_blocks_x = static_cast<int>((num_roots * (num_phi + num_branches) - 1) / num_threads_x) + 1;

	blocks.x = num_blocks_x;
	blocks.y = num_blocks_y;
	blocks.z = num_blocks_z;
	threads.x = num_threads_x;
	threads.y = num_threads_y;
	threads.z = num_threads_z;

	std::cout << "\nTransposing critical curve array...\n";
	starttime = std::chrono::high_resolution_clock::now();
	transpose_array_kernel<dtype> <<<blocks, threads>>> (ccs_init, (num_phi + num_branches), num_roots, ccs);
	if (cuda_error("transpose_array_kernel", true, __FILE__, __LINE__)) return -1;
	endtime = std::chrono::high_resolution_clock::now();
	std::cout << "Done transposing critical curve array. Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0 << " seconds.\n";

	std::cout << "\nFinding caustic positions...\n";
	starttime = std::chrono::high_resolution_clock::now();
	if (rectangular)
	{
		if (approx)
		{
			find_caustics_kernel<dtype> <<<blocks, threads>>> (ccs, (num_phi + num_branches)* num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, taylor, caustics);
		}
		else
		{
			find_caustics_kernel<dtype> <<<blocks, threads>>> (ccs, (num_phi + num_branches)* num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, caustics);
		}
	}
	else
	{
		find_caustics_kernel<dtype> <<<blocks, threads>>> (ccs, (num_phi + num_branches)* num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, caustics);
	}
	if (cuda_error("find_caustics_kernel", true, __FILE__, __LINE__)) return -1;
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
	outfile << "shear " << shear << "\n";
	outfile << "mu_ave " << mu_ave << "\n";
	outfile << "theta_e " << theta_e << "\n";
	outfile << "kappa_star " << kappa_star << "\n";
	outfile << "lower_mass_cutoff " << m_lower << "\n";
	outfile << "upper_mass_cutoff " << m_upper << "\n";
	outfile << "mean_mass " << mean_mass << "\n";
	outfile << "mean_squared_mass " << mean_squared_mass << "\n";
	outfile << "num_stars " << num_stars << "\n";
	if (rectangular)
	{
		outfile << "corner_x1 " << c.re << "\n";
		outfile << "corner_x2 " << c.im << "\n";
		if (approx)
		{
			outfile << "taylor " << taylor << "\n";
		}
	}
	else
	{
		outfile << "rad " << rad << "\n";
	}
	outfile << "num_roots " << num_roots << "\n";
	outfile << "num_phi " << num_phi << "\n";
	outfile << "num_branches " << num_branches << "\n";
	outfile << "random_seed " << random_seed << "\n";
	outfile << "max_error_1/mu " << max_error << "\n";
	outfile << "t_init_roots " << t_init_roots << "\n";
	outfile << "t_ccs " << t_ccs << "\n";
	outfile << "t_caustics " << t_caustics << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << outfile_prefix << "ccf_parameter_info.txt\n";

	std::cout << "\nWriting star info...\n";
	if (!write_star_file<dtype>(stars, num_stars, outfile_prefix + "ccf_stars" + outfile_type))
	{
		std::cerr << "Error. Unable to write star info to file " << outfile_prefix << "ccf_stars" + outfile_type << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << outfile_prefix << "ccf_stars" + outfile_type << "\n";


	/*write critical curve positions*/
	std::cout << "\nWriting critical curve positions...\n";
	if (outfile_type == ".txt")
	{
		if (!write_re_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_ccs_x" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs x info to file " << outfile_prefix << "ccf_ccs_x" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve x positions to file " << outfile_prefix << "ccf_ccs_x" + outfile_type << "\n";
		if (!write_im_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_ccs_y" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs y info to file " << outfile_prefix << "ccf_ccs_y" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve y positions to file " << outfile_prefix << "ccf_ccs_y" + outfile_type << "\n";
	}
	else
	{
		if (!write_complex_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_ccs" + outfile_type))
		{
			std::cerr << "Error. Unable to write ccs info to file " << outfile_prefix << "ccf_ccs" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve positions to file " << outfile_prefix << "ccf_ccs" + outfile_type << "\n";
	}


	/*write caustic positions*/
	std::cout << "\nWriting caustic positions...\n";
	if (outfile_type == ".txt")
	{
		if (!write_re_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_caustics_x" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic x info to file " << outfile_prefix << "ccf_caustics_x" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic x positions to file " << outfile_prefix << "ccf_caustics_x" + outfile_type << "\n";
		if (!write_im_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_caustics_y" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic y info to file " << outfile_prefix << "ccf_caustics_y" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic y positions to file " << outfile_prefix << "ccf_caustics_y" + outfile_type << "\n";
	}
	else
	{
		if (!write_complex_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, outfile_prefix + "ccf_caustics" + outfile_type))
		{
			std::cerr << "Error. Unable to write caustic info to file " << outfile_prefix << "ccf_caustics" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing caustic positions to file " << outfile_prefix << "ccf_caustics" + outfile_type << "\n";
	}

	std::cout << "\nDone.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

