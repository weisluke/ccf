/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "complex.cuh"
#include "ccf_microlensing.cuh"
#include "ccf_read_write_files.cuh"
#include "mass_function.cuh"
#include "star.cuh"
#include "stopwatch.cuh"
#include "util.hpp"

#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>


using dtype = double;

/******************************************************************************
constants to be used
******************************************************************************/
const dtype PI = 3.1415926535898;
constexpr int OPTS_SIZE = 2 * 21;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-k", "--kappa_tot",
	"-y", "--shear",
	"-s", "--smooth_fraction",
	"-ks", "--kappa_star",
	"-t", "--theta_e",
	"-mf", "--mass_function",
	"-ms", "--m_solar",
	"-ml", "--m_lower",
	"-mh", "--m_upper",
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
const std::map<std::string, enumMassFunction> MASS_FUNCTIONS{
	{"equal", equal},
	{"uniform", uniform},
	{"salpeter", salpeter},
	{"kroupa", kroupa}
};


/******************************************************************************
default input option values
******************************************************************************/
bool verbose = false;
dtype kappa_tot = 0.3;
dtype shear = 0.3;
dtype smooth_fraction = 0.1;
dtype kappa_star = 0.27;
dtype theta_e = 1;
std::string mass_function_str = "equal";
dtype m_solar = 1;
dtype m_lower = 0.01;
dtype m_upper = 50;
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



/******************************************************************************
Print the program usage help message

\param name -- name of the executable
******************************************************************************/
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
		<< "                                                                               \n"
		<< "Options:\n"
		<< "  -h,--help             Show this help message\n"
		<< "  -v,--verbose          Toggle verbose output. Takes no option value.\n"
		<< "  -k,--kappa_tot        Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "  -y,--shear            Specify the shear. Default value: " << shear << "\n"
		<< "  -s,--smooth_fraction  Specify the fraction of convergence due to smoothly\n"
		<< "                        distributed mass. Default value: " << smooth_fraction << "\n"
		<< "  -ks,--kappa_star      Specify the convergence in point mass lenses. If\n"
		<< "                        provided, this overrides any supplied value for the\n"
		<< "                        smooth fraction. Default value: " << kappa_star << "\n"
		<< "  -t,--theta_e          Specify the size of the Einstein radius of a unit mass\n"
		<< "                        point lens in arbitrary units. Default value: " << theta_e << "\n"
		<< "  -mf,--mass_function   Specify the mass function to use for the point mass\n"
		<< "                        lenses. Options are: equal, uniform, Salpeter, and\n"
		<< "                        Kroupa. Default value: " << mass_function_str << "\n"
		<< "  -ms,--m_solar         Specify the solar mass in arbitrary units.\n"
		<< "                        Default value: " << m_solar << "\n"
		<< "  -ml,--m_lower         Specify the lower mass cutoff in arbitrary units.\n"
		<< "                        Default value: " << m_lower << "\n"
		<< "  -mh,--m_upper         Specify the upper mass cutoff in arbitrary units.\n"
		<< "                        Default value: " << m_upper << "\n"
		<< "  -r,--rectangular      Specify whether the star field should be\n"
		<< "                        rectangular (1) or circular (0). Default value: " << rectangular << "\n"
		<< "  -a,--approx           Specify whether terms for alpha_smooth should be\n"
		<< "                        approximated (1) or exact (0). Default value: " << approx << "\n"
		<< "  -ts,--taylor          Specify the highest degree for the Taylor series of\n"
		<< "                        alpha_smooth. Default value: " << taylor << "\n"
		<< "  -ns,--num_stars       Specify the number of stars desired. Default value: " << num_stars << "\n"
		<< "                        All stars are taken to be of unit mass. If a range of\n"
		<< "                        masses are desired, please input them through a file as\n"
		<< "                        described in the -sf option.\n"
		<< "  -sf,--starfile        Specify the location of a star positions and masses\n"
		<< "                        file. The file may be either a whitespace delimited\n"
		<< "                        text file containing valid values for a star's x\n"
		<< "                        coordinate, y coordinate, and mass, in that order, on\n"
		<< "                        each line, or a binary file of star structures (as\n"
		<< "                        defined in this source code). If specified, the number\n"
		<< "                        of stars is determined through this file and the -ns\n"
		<< "                        option is ignored.\n"
		<< "  -np,--num_phi         Specify the number of steps used to vary phi in the\n"
		<< "                        range [0, 2*pi]. Default value: " << num_phi << "\n"
		<< "  -nb,--num_branches    Specify the number of branches to use for phi in the\n"
		<< "                        range [0, 2*pi]. Default value: " << num_branches << "\n"
		<< "  -rs,--random_seed     Specify the random seed for star field generation. A\n"
		<< "                        value of 0 is reserved for star input files.\n"
		<< "  -ot,--outfile_type    Specify the type of file to be output. Valid options\n"
		<< "                        are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "  -o,--outfile_prefix   Specify the prefix to be used in output filenames.\n"
		<< "                        Default value: " << outfile_prefix << "\n"
		<< "                        Lines of .txt output files are whitespace delimited.\n"
		<< "                        Filenames are:\n"
		<< "                          ccf_parameter_info  various parameter values used in\n"
		<< "                                                calculations\n"
		<< "                          ccf_stars           the first item is num_stars\n"
		<< "                                                followed by binary\n"
		<< "                                                representations of the star\n"
		<< "                                                structures\n"
		<< "                          ccf_ccs             the first item is num_roots and\n"
		<< "                                                the second item is\n"
		<< "                                                num_phi / num_branches + 1\n"
		<< "                                                followed by binary\n"
		<< "                                                representations of the complex\n"
		<< "                                                critical curve values\n"
		<< "                          ccf_caustics        the first item is num_roots and\n"
		<< "                                                the second item is\n"
		<< "                                                num_phi / num_branches + 1\n"
		<< "                                                followed by binary\n"
		<< "                                                representations of the complex\n"
		<< "                                                caustic curve values\n";
}



int main(int argc, char* argv[])
{
	/******************************************************************************
	set precision for printing numbers to screen
	******************************************************************************/
	std::cout.precision(7);

	/******************************************************************************
	if help option has been input, display usage message
	******************************************************************************/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	if there are input options, but not an even number (since all options take a
	parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	if ((argc - 1) % 2 != 0 &&
		!(cmd_option_exists(argv, argv + argc, "-v") || cmd_option_exists(argv, argv + argc, "--verbose")))
	{
		std::cerr << "Error. Not enough values for options.\n";
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	check that all options given are valid. use step of 2 since all input options
	take parameters (assumed to be given immediately after the option). start at 1,
	since first array element, argv[0], is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	for (int i = 1; i < argc; i += 2)
	{
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			verbose = true;
			i--;
			continue;
		}
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
		/******************************************************************************
		account for possible verbose option, which is a toggle and takes no input
		******************************************************************************/
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			i--;
			continue;
		}

		cmdinput = cmd_option_value(argv, argv + argc, argv[i]);

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				set_param("kappa_tot", kappa_tot, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-y") || argv[i] == std::string("--shear"))
		{
			try
			{
				set_param("shear", shear, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--smooth_fraction"))
		{
			if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
			{
				continue;
			}
			try
			{
				set_param("smooth_fraction", smooth_fraction, std::stod(cmdinput), verbose);
				if (smooth_fraction < 0)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be >= 0\n";
					return -1;
				}
				else if (smooth_fraction >= 1)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be < 1\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid smooth_fraction input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_star"))
		{
			try
			{
				set_param("kappa_star", kappa_star, std::stod(cmdinput), verbose);
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
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_e"))
		{
			try
			{
				set_param("theta_e", theta_e, std::stod(cmdinput), verbose);
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
		else if (argv[i] == std::string("-mf") || argv[i] == std::string("--mass_function"))
		{
			set_param("mass_function", mass_function_str, make_lowercase(cmdinput), verbose);
			if (!MASS_FUNCTIONS.count(mass_function_str))
			{
				std::cerr << "Error. Invalid mass_function input. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			try
			{
				set_param("m_solar", m_solar, std::stod(cmdinput), verbose);
				if (m_solar < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_solar input. m_solar must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_solar input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ml") || argv[i] == std::string("--m_lower"))
		{
			try
			{
				set_param("m_lower", m_lower, std::stod(cmdinput), verbose);
				if (m_lower < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_lower input. m_lower must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_lower input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-mh") || argv[i] == std::string("--m_upper"))
		{
			try
			{
				set_param("m_upper", m_upper, std::stod(cmdinput), verbose);
				if (m_upper < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (m_upper > std::numeric_limits<dtype>::max())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be < " << std::numeric_limits<dtype>::max() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_upper input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-r") || argv[i] == std::string("--rectangular"))
		{
			try
			{
				set_param("rectangular", rectangular, std::stoi(cmdinput), verbose);
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
				set_param("approx", approx, std::stoi(cmdinput), verbose);
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
				set_param("taylor", taylor, std::stoi(cmdinput), verbose);
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
				set_param("num_stars", num_stars, std::stoi(cmdinput), verbose);
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
			set_param("starfile", starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-np") || argv[i] == std::string("--num_phi"))
		{
			try
			{
				set_param("num_phi", num_phi, std::stoi(cmdinput), verbose);
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
				set_param("num_branches", num_branches, std::stoi(cmdinput), verbose);
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
				set_param("random_seed", random_seed, std::stoi(cmdinput), verbose);
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
			set_param("outfile_type", outfile_type, make_lowercase(cmdinput), verbose);
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", outfile_prefix, cmdinput, verbose);
		}
	}

	std::cout << "\n";

	if (num_phi % (2 * num_branches) != 0)
	{
		std::cerr << "Error. Invalid num_phi input. num_phi must be a multiple of 2*num_branches\n";
		return -1;
	}

	if (m_lower >= m_upper)
	{
		std::cerr << "Error. m_lower must be less than m_upper.\n";
		return -1;
	}

	if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
	{
		set_param("smooth_fraction", smooth_fraction, 1 - kappa_star / kappa_tot, verbose, true);
	}
	else
	{
		set_param("kappa_star", kappa_star, (1 - smooth_fraction) * kappa_tot, verbose, true);
	}

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	check that a CUDA capable device is present
	******************************************************************************/
	int n_devices = 0;

	cudaGetDeviceCount(&n_devices);
	if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return -1;

	if (verbose)
	{
		std::cout << "Available CUDA capable devices:\n\n";

		for (int i = 0; i < n_devices; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return -1;

			show_device_info(i, prop);
			std::cout << "\n";
		}
	}

	if (n_devices > 1)
	{
		std::cout << "More than one CUDA capable device detected. Defaulting to first device.\n\n";
	}
	cudaSetDevice(0);
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;


	/******************************************************************************
	determine mass function, <m>, and <m^2>
	******************************************************************************/
	enumMassFunction mass_function = MASS_FUNCTIONS.at(mass_function_str);
	dtype mean_mass;
	set_param("mean_mass", mean_mass, MassFunction<dtype>(mass_function).mean_mass(m_solar, m_lower, m_upper), verbose);
	dtype mean_mass2;
	set_param("mean_mass2", mean_mass2, MassFunction<dtype>(mass_function).mean_mass2(m_solar, m_lower, m_upper), verbose, true);

	/******************************************************************************
	calculated values for kappa_star, upper and lower mass cutoffs, <m>, and <m^2>
	******************************************************************************/
	dtype kappa_star_actual = static_cast<dtype>(1);
	dtype m_lower_actual = static_cast<dtype>(1);
	dtype m_upper_actual = static_cast<dtype>(1);
	dtype mean_mass_actual = static_cast<dtype>(1);
	dtype mean_mass2_actual = static_cast<dtype>(1);


	/******************************************************************************
	if star file is specified, check validity of values and set num_stars,
	m_lower_actual, m_upper_actual, mean_mass_actual, and mean_mass2_actual based
	on star information
	******************************************************************************/
	if (starfile != "")
	{
		std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

		if (!read_star_params<dtype>(num_stars, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n\n";

		m_lower = m_lower_actual;
		m_upper = m_upper_actual;
		mean_mass = mean_mass_actual;
		mean_mass2 = mean_mass2_actual;
	}

	/******************************************************************************
	average magnification of the system
	******************************************************************************/
	dtype mu_ave;
	set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose, true);

	std::cout << "Number of stars used: " << num_stars << "\n\n";

	Complex<dtype> c;
	set_param("corner", c,
		std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
		* Complex<dtype>(
			std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
			std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
			),
		verbose && rectangular);

	dtype rad;
	set_param("rad", rad, std::sqrt(theta_e * theta_e * num_stars * mean_mass / kappa_star), verbose && !rectangular);

	/******************************************************************************
	number of roots to be found
	******************************************************************************/
	int num_roots;
	set_param("num_roots", num_roots, 2 * num_stars, verbose, true && !(rectangular && approx));
	if (rectangular && approx)
	{
		set_param("num_roots", num_roots, num_roots + static_cast<int>(taylor / 2) * 2, verbose, true);
	}


	/******************************************************************************
	BEGIN memory allocation
	******************************************************************************/

	std::cout << "Beginning memory allocation...\n";

	curandState* states = nullptr;
	star<dtype>* stars = nullptr;
	Complex<dtype>* ccs_init = nullptr;
	Complex<dtype>* ccs = nullptr;
	bool* fin = nullptr;
	dtype* errs = nullptr;
	int* has_nan = nullptr;
	Complex<dtype>* caustics = nullptr;

	/******************************************************************************
	allocate memory for stars
	******************************************************************************/
	cudaMallocManaged(&states, num_stars * sizeof(curandState));
	if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&stars, num_stars * sizeof(star<dtype>));
	if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for array of critical curve positions
	******************************************************************************/
	cudaMallocManaged(&ccs_init, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*ccs_init)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for array of transposed critical curve positions
	******************************************************************************/
	cudaMallocManaged(&ccs, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*ccs)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	array to hold t/f values of whether or not roots have been found to desired
	precision
	******************************************************************************/
	cudaMallocManaged(&fin, num_branches * 2 * num_roots * sizeof(bool));
	if (cuda_error("cudaMallocManaged(*fin)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	array to hold root errors
	******************************************************************************/
	cudaMallocManaged(&errs, (num_phi + num_branches) * num_roots * sizeof(dtype));
	if (cuda_error("cudaMallocManaged(*errs)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	variable to hold whether array of root errors has nan errors or not
	******************************************************************************/
	cudaMallocManaged(&has_nan, sizeof(int));
	if (cuda_error("cudaMallocManaged(*has_nan)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	array to hold caustic positions
	******************************************************************************/
	cudaMallocManaged(&caustics, (num_phi + num_branches) * num_roots * sizeof(Complex<dtype>));
	if (cuda_error("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return -1;

	std::cout << "Done allocating memory.\n\n";

	/******************************************************************************
	END memory allocation
	******************************************************************************/


	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	number of threads per block, and number of blocks per grid
	uses 512 for number of threads in x dimension, as 1024 is the maximum allowable
	number of threads per block but is too large for some memory allocation, and
	512 is next power of 2 smaller
	******************************************************************************/
	set_threads(threads, 512);
	set_blocks(threads, blocks, num_stars);


	/******************************************************************************
	BEGIN populating star array
	******************************************************************************/

	if (starfile == "")
	{
		std::cout << "Generating star field...\n";

		/******************************************************************************
		if random seed was not provided, get one based on the time
		******************************************************************************/
		while (random_seed == 0)
		{
			set_param("random_seed", random_seed, static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()), verbose);
		}

		/******************************************************************************
		generate random star field if no star file has been given
		******************************************************************************/
		initialize_curand_states_kernel<dtype> <<<blocks, threads>>> (states, num_stars, random_seed);
		if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return -1;
		if (rectangular)
		{
			generate_rectangular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, c, mass_function, m_solar, m_lower, m_upper);
		}
		else
		{
			generate_circular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, rad, mass_function, m_solar, m_lower, m_upper);
		}
		if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return -1;

		std::cout << "Done generating star field.\n\n";

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		and mean_mass2_actual based on star information
		******************************************************************************/
		calculate_star_params<dtype>(stars, num_stars, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual);
		if (rectangular)
		{
			kappa_star_actual = PI * theta_e * theta_e * num_stars * mean_mass_actual / (4 * c.re * c.im);
		}
		else
		{
			kappa_star_actual = theta_e * theta_e * num_stars * mean_mass_actual / (rad * rad);
		}
	}
	else
	{
		/******************************************************************************
		ensure random seed is 0 to denote that stars come from external file
		******************************************************************************/
		set_param("random_seed", random_seed, 0, verbose);

		std::cout << "Reading star field from file " << starfile << "\n";

		/******************************************************************************
		reading star field from external file
		******************************************************************************/
		if (!read_star_file<dtype>(stars, num_stars, starfile))
		{
			std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done reading star field from file " << starfile << "\n\n";
	}

	/************************
	END populating star array
	************************/


	/******************************************************************************
	redefine thread and block size to maximize parallelization
	******************************************************************************/
	set_threads(threads, 32);
	set_blocks(threads, blocks, num_roots, 2, num_branches);


	/******************************************************************************
	set boolean (int) of errors having nan values to false (0)
	******************************************************************************/
	*has_nan = 0;

	/******************************************************************************
	initialize roots for centers of all branches to lie at starpos +/- 1
	******************************************************************************/
	print_verbose("Initializing root positions...\n", verbose);
	for (int j = 0; j < num_branches; j++)
	{
		int center = (num_phi / (2 * num_branches) + j * num_phi / num_branches + j) * num_roots;
		for (int i = 0; i < num_stars; i++)
		{
			ccs_init[center + i] = stars[i].position + 1;
			ccs_init[center + i + num_stars] = stars[i].position - 1;
		}
		if (rectangular && approx)
		{
			int nroots_extra = static_cast<int>(taylor / 2) * 2;
			for (int i = 0; i < nroots_extra; i++)
			{
				ccs_init[center + 2 * num_stars + i] = c.abs() * 
					Complex<dtype>(std::cos(2 * PI / nroots_extra * i), std::sin(2 * PI / nroots_extra * i));
			}
		}
	}
	print_verbose("Done initializing root positions.\n\n", verbose);

	/******************************************************************************
	initialize values of whether roots have been found to false
	twice the number of roots for a single value of phi for each branch, times the
	number of branches, because we will be growing roots for two values of phi
	simultaneously for each branch
	******************************************************************************/
	for (int i = 0; i < num_branches * 2 * num_roots; i++)
	{
		fin[i] = false;
	}

	for (int i = 0; i < (num_phi + num_branches) * num_roots; i++)
	{
		errs[i] = static_cast<dtype>(0);
	}

	/******************************************************************************
	number of iterations to use for root finding
	empirically, 30 seems to be roughly the amount needed
	******************************************************************************/
	int num_iters = 30;


	/******************************************************************************
	begin finding initial roots and calculate time taken in seconds
	******************************************************************************/
	std::cout << "Finding initial roots...\n";
	stopwatch.start();

	/******************************************************************************
	each iteration of this loop calculates updated positions of all roots for the
	center of each branch in parallel
	ideally, the number of loop iterations is enough to ensure that all roots are
	found to the desired accuracy
	******************************************************************************/
	for (int i = 0; i < num_iters; i++)
	{
		/******************************************************************************
		display percentage done
		******************************************************************************/
		print_progress(i, num_iters - 1);

		find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 
			rectangular, c, approx, taylor, ccs_init, num_roots, 0, num_phi, num_branches, fin);
		if (cuda_error("find_critical_curve_roots_kernel", true, __FILE__, __LINE__)) return -1;
	}
	double t_init_roots = stopwatch.stop();
	std::cout << "\nDone finding roots. Elapsed time: " << t_init_roots << " seconds.\n";


	/******************************************************************************
	calculate errors in 1/mu for initial roots
	******************************************************************************/
	print_verbose("Calculating maximum errors in 1/mu...\n", verbose);
	find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 
		rectangular, c, approx, taylor, 0, num_phi, num_branches, errs);
	if (cuda_error("find_errors_kernel", false, __FILE__, __LINE__)) return -1;

	has_nan_err_kernel<dtype> <<<blocks, threads>>> (errs, (num_phi + num_branches) * num_roots, has_nan);
	if (cuda_error("has_nan_err_kernel", true, __FILE__, __LINE__)) return -1;

	if (*has_nan)
	{
		std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
		return -1;
	}

	/******************************************************************************
	find max error and print
	must be performed in loops as CUDA does not currently have an atomicMax for
	floats or doubles, only ints
	******************************************************************************/
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
	print_verbose("Done calculating maximum errors in 1/mu.\n", verbose);
	std::cout << "Maximum error in 1/mu: " << max_error << "\n\n";


	/******************************************************************************
	reduce number of iterations needed, as roots should stay close to previous
	positions
	******************************************************************************/
	num_iters = 20;


	/******************************************************************************
	begin finding critical curves and calculate time taken in seconds
	******************************************************************************/
	std::cout << "Finding critical curve positions...\n";
	stopwatch.start();

	/******************************************************************************
	the outer loop will step through different values of phi
	we use num_phi/(2*num_branches) steps, as we will be working our way out from
	the middle of each branch for the array of roots simultaneously
	******************************************************************************/
	for (int j = 1; j <= num_phi / (2 * num_branches); j++)
	{
		/******************************************************************************
		set critical curve array elements to be equal to last roots
		fin array is reused each time
		******************************************************************************/
		prepare_roots_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, j, num_phi, num_branches, fin);
		if (cuda_error("prepare_roots_kernel", false, __FILE__, __LINE__)) return -1;

		/******************************************************************************
		calculate roots for current values of j
		******************************************************************************/
		for (int i = 0; i < num_iters; i++)
		{
			find_critical_curve_roots_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 
				rectangular, c, approx, taylor, ccs_init, num_roots, j, num_phi, num_branches, fin);
			if (cuda_error("find_critical_curve_roots_kernel", false, __FILE__, __LINE__)) return -1;
		}
		/******************************************************************************
		only perform synchronization call after roots have all been found
		this allows the print_progress call in the outer loop to accurately display the
		amount of work done so far
		one could move the synchronization call outside of the outer loop for a slight
		speed-up, at the cost of not knowing how far along in the process the
		computations have gone
		******************************************************************************/
		if (j * 100 / (num_phi / (2 * num_branches)) > (j - 1) * 100 / (num_phi / (2 * num_branches)))
		{
			cudaDeviceSynchronize();
			if (cuda_error("cudaDeviceSynchronize", false, __FILE__, __LINE__)) return -1;
			print_progress(j, num_phi / (2 * num_branches));
		}
	}
	double t_ccs = stopwatch.stop();
	std::cout << "\nDone finding critical curve positions. Elapsed time: " << t_ccs << " seconds.\n\n";


	/******************************************************************************
	find max error in 1/mu over whole critical curve array and print
	******************************************************************************/
	std::cout << "Finding maximum error in 1/mu over all calculated critical curve positions...\n";

	for (int j = 0; j <= num_phi / (2 * num_branches); j++)
	{
		find_errors_kernel<dtype> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 
			rectangular, c, approx, taylor, j, num_phi, num_branches, errs);
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
	std::cout << "Maximum error in 1/mu: " << max_error << "\n\n";


	/******************************************************************************
	redefine thread and block size to maximize parallelization
	******************************************************************************/
	set_threads(threads, 512);
	set_blocks(threads, blocks, num_roots * (num_phi + num_branches));

	print_verbose("Transposing critical curve array...\n", verbose);
	stopwatch.start();
	transpose_array_kernel<dtype> <<<blocks, threads>>> (ccs_init, (num_phi + num_branches), num_roots, ccs);
	if (cuda_error("transpose_array_kernel", true, __FILE__, __LINE__)) return -1;
	print_verbose("Done transposing critical curve array. Elapsed time: " + std::to_string(stopwatch.stop()) + " seconds.\n\n", verbose);
	
	std::cout << "Finding caustic positions...\n";
	stopwatch.start();
	find_caustics_kernel<dtype> <<<blocks, threads>>> (ccs, (num_phi + num_branches) * num_roots, kappa_tot, shear, theta_e, stars, num_stars, kappa_star, 
		rectangular, c, approx, taylor, caustics);
	if (cuda_error("find_caustics_kernel", true, __FILE__, __LINE__)) return -1;
	double t_caustics = stopwatch.stop();
	std::cout << "Done finding caustic positions. Elapsed time: " << t_caustics << " seconds.\n\n";



	/******************************************************************************
	stream for writing output files
	set precision to 9 digits
	******************************************************************************/
	std::ofstream outfile;
	outfile.precision(9);
	std::string fname;


	std::cout << "Writing parameter info...\n";
	fname = outfile_prefix + "ccf_parameter_info.txt";
	outfile.open(fname);
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return -1;
	}
	outfile << "kappa_tot " << kappa_tot << "\n";
	outfile << "shear " << shear << "\n";
	outfile << "mu_ave " << mu_ave << "\n";
	outfile << "smooth_fraction " << smooth_fraction << "\n";
	outfile << "kappa_star " << kappa_star << "\n";
	if (starfile == "")
	{
		outfile << "kappa_star_actual " << kappa_star_actual << "\n";
	}
	outfile << "theta_e " << theta_e << "\n";
	if (starfile == "")
	{
		outfile << "mass_function " << mass_function_str << "\n";
		if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
		{
			outfile << "m_solar " << m_solar << "\n";
		}
		if (mass_function_str != "equal")
		{
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "m_lower_actual " << m_lower_actual << "\n";
			outfile << "m_upper_actual " << m_upper_actual << "\n";
		}
		outfile << "mean_mass " << mean_mass << "\n";
		outfile << "mean_mass2 " << mean_mass2 << "\n";
		if (mass_function_str != "equal")
		{
			outfile << "mean_mass_actual " << mean_mass_actual << "\n";
			outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		}
	}
	else
	{
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
	}
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
	std::cout << "Done writing parameter info to file " << fname << "\n\n";


	std::cout << "Writing star info...\n";
	fname = outfile_prefix + "ccf_stars" + outfile_type;
	if (!write_star_file<dtype>(stars, num_stars, fname))
	{
		std::cerr << "Error. Unable to write star info to file " << fname << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << fname << "\n\n";


	/******************************************************************************
	write critical curve positions
	******************************************************************************/
	std::cout << "Writing critical curve positions...\n";
	if (outfile_type == ".txt")
	{
		fname = outfile_prefix + "ccf_ccs_x" + outfile_type;
		if (!write_re_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write ccs x info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve x positions to file " << fname << "\n";

		fname = outfile_prefix + "ccf_ccs_y" + outfile_type;
		if (!write_im_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write ccs y info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve y positions to file " << fname << "\n";
	}
	else
	{
		fname = outfile_prefix + "ccf_ccs" + outfile_type;
		if (!write_complex_array<dtype>(ccs, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write ccs info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing critical curve positions to file " << fname << "\n";
	}
	std::cout << "\n";


	/******************************************************************************
	write caustic positions
	******************************************************************************/
	std::cout << "Writing caustic positions...\n";
	if (outfile_type == ".txt")
	{
		fname = outfile_prefix + "ccf_caustics_x" + outfile_type;
		if (!write_re_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write caustic x info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing caustic x positions to file " << fname << "\n";

		fname = outfile_prefix + "ccf_caustics_y" + outfile_type;
		if (!write_im_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write caustic y info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing caustic y positions to file " << fname << "\n";
	}
	else
	{
		fname = outfile_prefix + "ccf_caustics" + outfile_type;
		if (!write_complex_array<dtype>(caustics, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write caustic info to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing caustic positions to file " << fname << "\n";
	}
	std::cout << "\n";

	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

