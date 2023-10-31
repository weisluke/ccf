/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "ccf.cuh"
#include "util.hpp"

#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


using dtype = double; //type to be used throughout this program. int, float, or double
CCF<dtype> ccf;

/******************************************************************************
constants to be used
******************************************************************************/
constexpr int OPTS_SIZE = 2 * 19;
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
	"-ns", "--num_stars",
	"-sf", "--starfile",
	"-np", "--num_phi",
	"-nb", "--num_branches",
	"-rs", "--random_seed",
	"-o", "--outfile_prefix"
};

/******************************************************************************
default input option values
******************************************************************************/
bool verbose = false;



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
		<< "  -k,--kappa_tot        Specify the total convergence. Default value: " << ccf.kappa_tot << "\n"
		<< "  -y,--shear            Specify the shear. Default value: " << ccf.shear << "\n"
		<< "  -s,--smooth_fraction  Specify the fraction of convergence due to smoothly\n"
		<< "                        distributed mass. Default value: " << ccf.smooth_fraction << "\n"
		<< "  -ks,--kappa_star      Specify the convergence in point mass lenses. If\n"
		<< "                        provided, this overrides any supplied value for the\n"
		<< "                        smooth fraction. Default value: " << ccf.kappa_star << "\n"
		<< "  -t,--theta_e          Specify the size of the Einstein radius of a unit mass\n"
		<< "                        point lens in arbitrary units. Default value: " << ccf.theta_e << "\n"
		<< "  -mf,--mass_function   Specify the mass function to use for the point mass\n"
		<< "                        lenses. Options are: equal, uniform, Salpeter, and\n"
		<< "                        Kroupa. Default value: " << ccf.mass_function_str << "\n"
		<< "  -ms,--m_solar         Specify the solar mass in arbitrary units.\n"
		<< "                        Default value: " << ccf.m_solar << "\n"
		<< "  -ml,--m_lower         Specify the lower mass cutoff in arbitrary units.\n"
		<< "                        Default value: " << ccf.m_lower << "\n"
		<< "  -mh,--m_upper         Specify the upper mass cutoff in arbitrary units.\n"
		<< "                        Default value: " << ccf.m_upper << "\n"
		<< "  -r,--rectangular      Specify whether the star field should be\n"
		<< "                        rectangular (1) or circular (0). Default value: " << ccf.rectangular << "\n"
		<< "  -a,--approx           Specify whether terms for alpha_smooth should be\n"
		<< "                        approximated (1) or exact (0). Default value: " << ccf.approx << "\n"
		<< "  -ns,--num_stars       Specify the number of stars desired. Default value: " << ccf.num_stars << "\n"
		<< "                        All stars are taken to be of unit mass. If a range of\n"
		<< "                        masses are desired, please input them through a file as\n"
		<< "                        described in the -sf option.\n"
		<< "  -sf,--starfile        Specify the location of a binary file containing values\n"
		<< "                        for num_stars, rectangular, corner, theta_e, and the\n"
		<< "                        star positions and masses, in an order as defined in\n"
		<< "                        this source code.\n"
		<< "  -np,--num_phi         Specify the number of steps used to vary phi in the\n"
		<< "                        range [0, 2*pi]. Default value: " << ccf.num_phi << "\n"
		<< "  -nb,--num_branches    Specify the number of branches to use for phi in the\n"
		<< "                        range [0, 2*pi]. Default value: " << ccf.num_branches << "\n"
		<< "  -rs,--random_seed     Specify the random seed for star field generation. A\n"
		<< "                        value of 0 is reserved for star input files.\n"
		<< "  -o,--outfile_prefix   Specify the prefix to be used in output filenames.\n"
		<< "                        Default value: " << ccf.outfile_prefix << "\n";
}



int main(int argc, char* argv[])
{
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
		std::cerr << "Error. Invalid input syntax.\n";
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
				set_param("kappa_tot", ccf.kappa_tot, std::stod(cmdinput), verbose);
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
				set_param("shear", ccf.shear, std::stod(cmdinput), verbose);
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
				set_param("smooth_fraction", ccf.smooth_fraction, std::stod(cmdinput), verbose);
				if (ccf.smooth_fraction < 0)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be >= 0\n";
					return -1;
				}
				else if (ccf.smooth_fraction >= 1)
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
				set_param("kappa_star", ccf.kappa_star, std::stod(cmdinput), verbose);
				if (ccf.kappa_star < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid kappa_star input. kappa_star must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("theta_e", ccf.theta_e, std::stod(cmdinput), verbose);
				if (ccf.theta_e < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
			set_param("mass_function", ccf.mass_function_str, make_lowercase(cmdinput), verbose);
			if (!massfunctions::MASS_FUNCTIONS.count(ccf.mass_function_str))
			{
				std::cerr << "Error. Invalid mass_function input. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			try
			{
				set_param("m_solar", ccf.m_solar, std::stod(cmdinput), verbose);
				if (ccf.m_solar < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_solar input. m_solar must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("m_lower", ccf.m_lower, std::stod(cmdinput), verbose);
				if (ccf.m_lower < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_lower input. m_lower must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("m_upper", ccf.m_upper, std::stod(cmdinput), verbose);
				if (ccf.m_upper < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be >= " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (ccf.m_upper > std::numeric_limits<dtype>::max())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be <= " << std::numeric_limits<dtype>::max() << "\n";
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
				set_param("rectangular", ccf.rectangular, std::stoi(cmdinput), verbose);
				if (ccf.rectangular != 0 && ccf.rectangular != 1)
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
				set_param("approx", ccf.approx, std::stoi(cmdinput), verbose);
				if (ccf.approx != 0 && ccf.approx != 1)
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
		else if (argv[i] == std::string("-ns") || argv[i] == std::string("--num_stars"))
		{
			try
			{
				set_param("num_stars", ccf.num_stars, std::stoi(cmdinput), verbose);
				if (ccf.num_stars < 1)
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
			set_param("starfile", ccf.starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-np") || argv[i] == std::string("--num_phi"))
		{
			try
			{
				set_param("num_phi", ccf.num_phi, std::stoi(cmdinput), verbose);
				if (ccf.num_phi < 1 || ccf.num_phi % 2 != 0)
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
				set_param("num_branches", ccf.num_branches, std::stoi(cmdinput), verbose);
				if (ccf.num_branches < 1)
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
			if (cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--star_file"))
			{
				continue;
			}
			try
			{
				set_param("random_seed", ccf.random_seed, std::stoi(cmdinput), verbose);
				if (ccf.random_seed == 0)
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
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", ccf.outfile_prefix, cmdinput, verbose);
		}
	}

	if (ccf.num_phi % (2 * ccf.num_branches) != 0)
	{
		std::cerr << "Error. Invalid num_phi input. num_phi must be a multiple of 2*num_branches\n";
		return -1;
	}

	if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
	{
		set_param("smooth_fraction", ccf.smooth_fraction, 1 - ccf.kappa_star / ccf.kappa_tot, verbose);
	}
	else
	{
		set_param("kappa_star", ccf.kappa_star, (1 - ccf.smooth_fraction) * ccf.kappa_tot, verbose);
	}

	if (ccf.mass_function_str == "equal")
	{
		set_param("m_lower", ccf.m_lower, 1, verbose);
		set_param("m_upper", ccf.m_upper, 1, verbose);
	}
	else if (ccf.m_lower > ccf.m_upper)
	{
		std::cerr << "Error. m_lower must be <= m_upper.\n";
		return -1;
	}

	std::cout << "\n";

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
	run and save files
	******************************************************************************/
	if (!ccf.run(verbose)) return -1;
	if (!ccf.save(verbose)) return -1;

	
	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

