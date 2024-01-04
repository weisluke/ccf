#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "ccf_functions.cuh"
#include "fmm.cuh"
#include "mass_function.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util.cuh"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>
#include <vector>


template <typename T>
class CCF
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	T kappa_tot = static_cast<T>(0.3);
	T shear = static_cast<T>(0.3);
	T kappa_star = static_cast<T>(0.27);
	T theta_e = static_cast<T>(1);
	std::string mass_function_str = "equal";
	T m_solar = static_cast<T>(1);
	T m_lower = static_cast<T>(0.01);
	T m_upper = static_cast<T>(50);
	int rectangular = 1;
	int approx = 1;
	int num_stars = 137;
	std::string starfile = "";
	int num_phi = 100;
	int num_branches = 1;
	int random_seed = 0;
	std::string outfile_prefix = "./";


	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	CCF()
	{

	}


private:
	/******************************************************************************
	constant variables
	******************************************************************************/
	const T PI = static_cast<T>(3.1415926535898);
	const T E = static_cast<T>(2.718281828459);
	const std::string outfile_type = ".bin";

	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;
	double t_init_roots;
	double t_ccs;
	double t_caustics;

	/******************************************************************************
	derived variables
	******************************************************************************/
	massfunctions::massfunction mass_function;
	T mean_mass;
	T mean_mass2;

	T kappa_star_actual;
	T m_lower_actual;
	T m_upper_actual;
	T mean_mass_actual;
	T mean_mass2_actual;

	T mu_ave;
	Complex<T> corner;
	int taylor_smooth;

	T alpha_error; //error in the deflection angle

	int expansion_order;

	T root_half_length;
	int tree_levels = 0;
	std::vector<TreeNode<T>*> tree = {};
	std::vector<int> num_nodes = {};

	int num_roots;
	T max_error;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states = nullptr;
	star<T>* stars = nullptr;
	star<T>* temp_stars = nullptr;

	int* binomial_coeffs = nullptr;

	Complex<T>* ccs_init = nullptr;
	Complex<T>* ccs = nullptr;
	bool* fin = nullptr;
	T* errs = nullptr;
	int* has_nan = nullptr;
	Complex<T>* caustics = nullptr;



	bool check_input_params(bool verbose)
	{
		std::cout << "Checking input parameters...\n";


		if (kappa_tot < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_tot must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (kappa_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		if (kappa_star > kappa_tot)
		{
			std::cerr << "Error. kappa_star must be <= kappa_tot\n";
			return false;
		}

		if (theta_e < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. theta_e must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (!massfunctions::MASS_FUNCTIONS.count(mass_function_str))
		{
			std::cerr << "Error. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
			return false;
		}

		if (m_solar < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_solar must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (m_lower < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_lower must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (m_upper < m_lower)
		{
			std::cerr << "Error. m_upper must be >= m_lower.\n";
			return false;
		}

		if (rectangular != 0 && rectangular != 1)
		{
			std::cerr << "Error. rectangular must be 1 (rectangular) or 0 (circular).\n";
			return false;
		}

		if (approx != 0 && approx != 1)
		{
			std::cerr << "Error. approx must be 1 (approximate) or 0 (exact).\n";
			return false;
		}
		
		if (num_stars < 1)
		{
			std::cerr << "Error. num_stars must be an integer > 0\n";
			return false;
		}

		if (num_phi < 1 || num_phi % 2 != 0)
		{
			std::cerr << "Error. num_phi must be an even integer > 0\n";
			return false;
		}

		if (num_branches < 1)
		{
			std::cerr << "Error. num_branches must be an integer > 0\n";
			return false;
		}

		if (num_phi % (2 * num_branches) != 0)
		{
			std::cerr << "Error. num_phi must be a multiple of 2 * num_branches\n";
			return -1;
		}


		std::cout << "Done checking input parameters.\n\n";
		
		return true;
	}
	
	bool calculate_derived_params(bool verbose)
	{
		std::cout << "Calculating derived parameters...\n";
		stopwatch.start();

		/******************************************************************************
		if star file is not specified, set the mass function, mean_mass, and
		mean_mass2
		******************************************************************************/
		if (starfile == "")
		{
			if (mass_function_str == "equal")
			{
				set_param("m_lower", m_lower, 1, verbose);
				set_param("m_upper", m_upper, 1, verbose);
			}

			/******************************************************************************
			determine mass function, <m>, and <m^2>
			******************************************************************************/
			mass_function = massfunctions::MASS_FUNCTIONS.at(mass_function_str);
			set_param("mean_mass", mean_mass, MassFunction<T>(mass_function).mean_mass(m_solar, m_lower, m_upper), verbose);
			set_param("mean_mass2", mean_mass2, MassFunction<T>(mass_function).mean_mass2(m_solar, m_lower, m_upper), verbose);
		}
		/******************************************************************************
		if star file is specified, check validity of values and set num_stars,
		rectangular, corner, theta_e, stars, kappa_star, m_lower, m_upper, mean_mass,
		and mean_mass2 based on star information
		******************************************************************************/
		else
		{
			std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

			if (!read_star_file<T>(num_stars, rectangular, corner, theta_e, stars,
				kappa_star, m_lower, m_upper, mean_mass, mean_mass2, starfile))
			{
				std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
				return false;
			}

			set_param("num_stars", num_stars, num_stars, verbose);
			set_param("rectangular", rectangular, rectangular, verbose);
			set_param("corner", corner, corner, verbose);
			set_param("theta_e", theta_e, theta_e, verbose);
			set_param("kappa_star", kappa_star, kappa_star, verbose);
			if (kappa_star > kappa_tot)
 			{
 				std::cerr << "Error. kappa_star must be <= kappa_tot\n";
 				return false;
 			}
			set_param("m_lower", m_lower, m_lower, verbose);
			set_param("m_upper", m_upper, m_upper, verbose);
			set_param("mean_mass", mean_mass, mean_mass, verbose);
			set_param("mean_mass2", mean_mass2, mean_mass2, verbose);

			std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n";
		}

		/******************************************************************************
		average magnification of the system
		******************************************************************************/
		set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

		/******************************************************************************
		if stars are not drawn from external file, calculate corner of the star field
		******************************************************************************/
		if (starfile == "")
		{
			if (rectangular)
			{
				corner = std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
					* Complex<T>(
						std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
						std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
					);
				set_param("corner", corner, corner, verbose);
			}
			else
			{
				corner = std::sqrt(theta_e * theta_e * num_stars * mean_mass / (kappa_star * 2 * ((1 - kappa_tot) * (1 - kappa_tot) + shear * shear)))
					* Complex<T>(
						std::abs(1 - kappa_tot - shear),
						std::abs(1 - kappa_tot + shear)
					);
				set_param("corner", corner, corner, verbose);
			}
		}

		alpha_error = theta_e * 0.0000001; //error is 10^-7 einstein radii

		taylor_smooth = std::max(
			static_cast<int>(std::log(2 * kappa_star * corner.abs() / (alpha_error * PI)) / std::log(1.1)),
			1);
		set_param("taylor_smooth", taylor_smooth, taylor_smooth, verbose && rectangular && approx);
		
		/******************************************************************************
		number of roots to be found
		******************************************************************************/
		set_param("num_roots", num_roots, 2 * num_stars, verbose && !(rectangular && approx));
		if (rectangular && approx)
		{
			set_param("num_roots", num_roots, num_roots + static_cast<int>(taylor_smooth / 2) * 2, verbose);
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool allocate_initialize_memory(bool verbose)
	{
		std::cout << "Allocating memory...\n";
		stopwatch.start();

		/******************************************************************************
		allocate memory for stars
		******************************************************************************/
		cudaMallocManaged(&states, num_stars * sizeof(curandState));
		if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return false;
		if (stars == nullptr) // if memory wasn't allocated already due to reading a star file
		{
			cudaMallocManaged(&stars, num_stars * sizeof(star<T>));
			if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
		}
		cudaMallocManaged(&temp_stars, num_stars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for binomial coefficients
		******************************************************************************/
		cudaMallocManaged(&binomial_coeffs, (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(int));
		if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for array of critical curve positions
		******************************************************************************/
		cudaMallocManaged(&ccs_init, (num_phi + num_branches) * num_roots * sizeof(Complex<T>));
		if (cuda_error("cudaMallocManaged(*ccs_init)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for array of transposed critical curve positions
		******************************************************************************/
		cudaMallocManaged(&ccs, (num_phi + num_branches) * num_roots * sizeof(Complex<T>));
		if (cuda_error("cudaMallocManaged(*ccs)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		array to hold t/f values of whether or not roots have been found to desired
		precision
		******************************************************************************/
		cudaMallocManaged(&fin, num_branches * 2 * num_roots * sizeof(bool));
		if (cuda_error("cudaMallocManaged(*fin)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		array to hold root errors
		******************************************************************************/
		cudaMallocManaged(&errs, (num_phi + num_branches) * num_roots * sizeof(T));
		if (cuda_error("cudaMallocManaged(*errs)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		variable to hold whether array of root errors has nan errors or not
		******************************************************************************/
		cudaMallocManaged(&has_nan, sizeof(int));
		if (cuda_error("cudaMallocManaged(*has_nan)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		array to hold caustic positions
		******************************************************************************/
		cudaMallocManaged(&caustics, (num_phi + num_branches) * num_roots * sizeof(Complex<T>));
		if (cuda_error("cudaMallocManaged(*caustics)", false, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		std::cout << "Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n\n";


		/******************************************************************************
		initialize values of whether roots have been found to false
		twice the number of roots for a single value of phi for each branch, times the
		number of branches, because we will be growing roots for two values of phi
		simultaneously for each branch
		******************************************************************************/
		
		std::cout << "Initializing array values...\n";
		stopwatch.start();

		for (int i = 0; i < num_branches * 2 * num_roots; i++)
		{
			fin[i] = false;
		}

		for (int i = 0; i < (num_phi + num_branches) * num_roots; i++)
		{
			errs[i] = static_cast<T>(0);
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done initializing array values. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool populate_star_array(bool verbose)
	{
		/******************************************************************************
		BEGIN populating star array
		******************************************************************************/

		set_threads(threads, 512);
		set_blocks(threads, blocks, num_stars);

		if (starfile == "")
		{
			std::cout << "Generating star field...\n";
			stopwatch.start();

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
			initialize_curand_states_kernel<T> <<<blocks, threads>>> (states, num_stars, random_seed);
			if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return false;
			generate_star_field_kernel<T> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, mass_function, m_solar, m_lower, m_upper);
			if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return false;

			t_elapsed = stopwatch.stop();
			std::cout << "Done generating star field. Elapsed time: " << t_elapsed << " seconds.\n\n";
		}
		else
		{
			/******************************************************************************
			ensure random seed is 0 to denote that stars come from external file
			******************************************************************************/
			set_param("random_seed", random_seed, 0, verbose);
		}

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		and mean_mass2_actual based on star information
		******************************************************************************/
		calculate_star_params<T>(num_stars, rectangular, corner, theta_e, stars,
			kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual);

		set_param("kappa_star_actual", kappa_star_actual, kappa_star_actual, verbose);
		set_param("m_lower_actual", m_lower_actual, m_lower_actual, verbose);
		set_param("m_upper_actual", m_upper_actual, m_upper_actual, verbose);
		set_param("mean_mass_actual", mean_mass_actual, mean_mass_actual, verbose);
		set_param("mean_mass2_actual", mean_mass2_actual, mean_mass2_actual, verbose, true);

		/******************************************************************************
		END populating star array
		******************************************************************************/


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
				int nroots_extra = static_cast<int>(taylor_smooth / 2) * 2;
				for (int i = 0; i < nroots_extra; i++)
				{
					ccs_init[center + 2 * num_stars + i] = corner.abs() *
						Complex<T>(std::cos(2 * PI / nroots_extra * i), std::sin(2 * PI / nroots_extra * i));
				}
			}
		}
		print_verbose("Done initializing root positions.\n\n", verbose);

		return true;
	}

	bool create_tree(bool verbose)
	{
		/******************************************************************************
		BEGIN create root node, then create children and sort stars
		******************************************************************************/

		if (rectangular)
		{
			root_half_length = corner.re > corner.im ? corner.re : corner.im;
		}
		else
		{
			root_half_length = corner.abs();
		}
		/******************************************************************************
		upscale root half length slightly to ensure it encompasses all stars
		******************************************************************************/
		set_param("root_half_length", root_half_length, 1.1 * root_half_length, verbose);

		/******************************************************************************
		push empty pointer into tree, add 1 to number of nodes, and allocate memory
		******************************************************************************/
		tree.push_back(nullptr);
		num_nodes.push_back(1);
		cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
		if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		initialize root node
		******************************************************************************/
		tree[0][0] = TreeNode<T>(Complex<T>(0, 0), root_half_length, 0);
		tree[0][0].numstars = num_stars;


		int* max_num_stars_in_level;
		int* min_num_stars_in_level;
		int* num_nonempty_nodes;
		cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&num_nonempty_nodes, sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;

		std::cout << "Creating children and sorting stars...\n";
		stopwatch.start();
		set_threads(threads, 512);
		do
		{
			print_verbose("\nProcessing level " + std::to_string(tree_levels) + "\n", verbose);

			*max_num_stars_in_level = 0;
			*min_num_stars_in_level = num_stars;
			*num_nonempty_nodes = 0;

			set_blocks(threads, blocks, num_nodes[tree_levels]);
			treenode::get_node_star_info_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels],
				num_nonempty_nodes, min_num_stars_in_level, max_num_stars_in_level);
			if (cuda_error("get_node_star_info_kernel", true, __FILE__, __LINE__)) return false;

			print_verbose("Maximum number of stars in a node and its neighbors is " + std::to_string(*max_num_stars_in_level) + "\n", verbose);
			print_verbose("Minimum number of stars in a node and its neighbors is " + std::to_string(*min_num_stars_in_level) + "\n", verbose);

			if (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT)
			{
				print_verbose("Number of non-empty children: " + std::to_string(*num_nonempty_nodes * 4) + "\n", verbose);

				print_verbose("Allocating memory for children...\n", verbose);
				tree.push_back(nullptr);
				num_nodes.push_back(*num_nonempty_nodes * 4);
				cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
				if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

				print_verbose("Creating children...\n", verbose);
				(*num_nonempty_nodes)--; //subtract one since value is size of array, and instead needs to be the first allocatable element
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::create_children_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], num_nonempty_nodes, tree[tree_levels + 1]);
				if (cuda_error("create_children_kernel", true, __FILE__, __LINE__)) return false;

				print_verbose("Sorting stars...\n", verbose);
				set_blocks(threads, blocks, 512 * num_nodes[tree_levels]);
				treenode::sort_stars_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], stars, temp_stars);
				if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return false;

				tree_levels++;

				print_verbose("Setting neighbors...\n", verbose);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::set_neighbors_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels]);
				if (cuda_error("set_neighbors_kernel", true, __FILE__, __LINE__)) return false;
			}
		} while (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT);
		set_param("tree_levels", tree_levels, tree_levels, verbose);

		t_elapsed = stopwatch.stop();
		std::cout << "Done creating children and sorting stars. Elapsed time: " << t_elapsed << " seconds.\n\n";

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		expansion_order = static_cast<int>(2 * std::log2(theta_e) - std::log2(root_half_length * alpha_error))
			 + tree_levels + 1;
		set_param("expansion_order", expansion_order, expansion_order, verbose, true);
		if (expansion_order < 3)
		{
			std::cerr << "Error. Expansion order needs to be >= 3\n";
			return false;
		}
		else if (expansion_order > treenode::MAX_EXPANSION_ORDER)
		{
			std::cerr << "Error. Maximum allowed expansion order is " << treenode::MAX_EXPANSION_ORDER << "\n";
			return false;
		}

		print_verbose("Calculating binomial coefficients...\n", verbose);
		calculate_binomial_coeffs(binomial_coeffs, 2 * expansion_order);
		print_verbose("Done calculating binomial coefficients.\n\n", verbose);


		/******************************************************************************
		BEGIN calculating multipole and local coefficients
		******************************************************************************/

		std::cout << "Calculating multipole and local coefficients...\n";
		stopwatch.start();

		set_threads(threads, 16, expansion_order + 1);
		set_blocks(threads, blocks, num_nodes[tree_levels], (expansion_order + 1));
		fmm::calculate_multipole_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[tree_levels], num_nodes[tree_levels], expansion_order, stars);

		set_threads(threads, 4, expansion_order + 1, 4);
		for (int i = tree_levels - 1; i >= 0; i--)
		{
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), 4);
			fmm::calculate_M2M_coeffs_kernel<T> <<<blocks, threads, 4 * 4 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}

		/******************************************************************************
		local coefficients are non zero only starting at the second level
		******************************************************************************/
		for (int i = 2; i <= tree_levels; i++)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_L2L_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 1, expansion_order + 1, 27);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), 27);
			fmm::calculate_M2L_coeffs_kernel<T> <<<blocks, threads, 1 * 27 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}
		if (cuda_error("calculate_coeffs_kernels", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		std::cout << "Done calculating multipole and local coefficients. Elapsed time: " << t_elapsed << " seconds.\n\n";
		
		/******************************************************************************
		END calculating multipole and local coefficients
		******************************************************************************/

		return true;
	}

	bool find_initial_roots(bool verbose)
	{
		/******************************************************************************
		number of iterations to use for root finding
		empirically, 30 seems to be roughly the amount needed
		******************************************************************************/
		int num_iters = 30;

		set_threads(threads, 32);
		set_blocks(threads, blocks, num_roots, 2, num_branches);

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

			find_critical_curve_roots_kernel<T> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, kappa_star, tree[0],
				rectangular, corner, approx, taylor_smooth, ccs_init, num_roots, 0, num_phi, num_branches, fin);
			if (cuda_error("find_critical_curve_roots_kernel", true, __FILE__, __LINE__)) return false;
		}
		t_init_roots = stopwatch.stop();
		std::cout << "\nDone finding roots. Elapsed time: " << t_elapsed << " seconds.\n";


		/******************************************************************************
		set boolean (int) of errors having nan values to false (0)
		******************************************************************************/
		*has_nan = 0;

		/******************************************************************************
		calculate errors in 1/mu for initial roots
		******************************************************************************/
		print_verbose("Calculating maximum errors in 1/mu...\n", verbose);
		find_errors_kernel<T> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, kappa_star, tree[0],
			rectangular, corner, approx, taylor_smooth, 0, num_phi, num_branches, errs);
		if (cuda_error("find_errors_kernel", false, __FILE__, __LINE__)) return false;

		has_nan_err_kernel<T> <<<blocks, threads>>> (errs, (num_phi + num_branches) * num_roots, has_nan);
		if (cuda_error("has_nan_err_kernel", true, __FILE__, __LINE__)) return false;

		if (*has_nan)
		{
			std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
			return false;
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
			max_err_kernel<T> <<<blocks, threads>>> (errs, num_errs);
			if (cuda_error("max_err_kernel", true, __FILE__, __LINE__)) return false;
		}
		max_error = errs[0];
		print_verbose("Done calculating maximum errors in 1/mu.\n", verbose);
		std::cout << "Maximum error in 1/mu: " << max_error << "\n\n";


		return true;
	}

	bool find_ccs_caustics(bool verbose)
	{
		/******************************************************************************
		reduce number of iterations needed, as roots should stay close to previous
		positions
		******************************************************************************/
		int num_iters = 20;

		set_threads(threads, 32);
		set_blocks(threads, blocks, num_roots, 2, num_branches);

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
			prepare_roots_kernel<T> <<<blocks, threads>>> (ccs_init, num_roots, j, num_phi, num_branches, fin);
			if (cuda_error("prepare_roots_kernel", false, __FILE__, __LINE__)) return false;

			/******************************************************************************
			calculate roots for current values of j
			******************************************************************************/
			for (int i = 0; i < num_iters; i++)
			{
				find_critical_curve_roots_kernel<T> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, kappa_star, tree[0],
					rectangular, corner, approx, taylor_smooth, ccs_init, num_roots, j, num_phi, num_branches, fin);
				if (cuda_error("find_critical_curve_roots_kernel", false, __FILE__, __LINE__)) return false;
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
				if (cuda_error("cudaDeviceSynchronize", false, __FILE__, __LINE__)) return false;
				print_progress(j, num_phi / (2 * num_branches));
			}
		}
		t_ccs = stopwatch.stop();
		std::cout << "\nDone finding critical curve positions. Elapsed time: " << t_ccs << " seconds.\n\n";


		/******************************************************************************
		set boolean (int) of errors having nan values to false (0)
		******************************************************************************/
		*has_nan = 0;

		/******************************************************************************
		find max error in 1/mu over whole critical curve array and print
		******************************************************************************/
		std::cout << "Finding maximum error in 1/mu over all calculated critical curve positions...\n";

		for (int j = 0; j <= num_phi / (2 * num_branches); j++)
		{
			find_errors_kernel<T> <<<blocks, threads>>> (ccs_init, num_roots, kappa_tot, shear, theta_e, stars, kappa_star, tree[0],
				rectangular, corner, approx, taylor_smooth, j, num_phi, num_branches, errs);
			if (cuda_error("find_errors_kernel", false, __FILE__, __LINE__)) return false;
		}

		has_nan_err_kernel<T> <<<blocks, threads>>> (errs, (num_phi + num_branches) * num_roots, has_nan);
		if (cuda_error("has_nan_err_kernel", true, __FILE__, __LINE__)) return false;

		if (*has_nan)
		{
			std::cerr << "Error. Errors in 1/mu contain values which are not positive real numbers.\n";
			return false;
		}

		int num_errs = (num_phi + num_branches) * num_roots;
		while (num_errs > 1)
		{
			if (num_errs % 2 != 0)
			{
				errs[num_errs - 2] = std::fmax(errs[num_errs - 2], errs[num_errs - 1]);
				num_errs -= 1;
			}
			num_errs /= 2;
			max_err_kernel<T> <<<blocks, threads>>> (errs, num_errs);
			if (cuda_error("max_err_kernel", true, __FILE__, __LINE__)) return false;
		}
		max_error = errs[0];
		std::cout << "Maximum error in 1/mu: " << max_error << "\n\n";


		set_threads(threads, 512);
		set_blocks(threads, blocks, num_roots * (num_phi + num_branches));

		print_verbose("Transposing critical curve array...\n", verbose);
		stopwatch.start();
		transpose_array_kernel<T> <<<blocks, threads>>> (ccs_init, (num_phi + num_branches), num_roots, ccs);
		if (cuda_error("transpose_array_kernel", true, __FILE__, __LINE__)) return false;
		t_elapsed = stopwatch.stop();
		print_verbose("Done transposing critical curve array. Elapsed time: " + std::to_string(t_elapsed) + " seconds.\n\n", verbose);

		std::cout << "Finding caustic positions...\n";
		stopwatch.start();
		find_caustics_kernel<T> <<<blocks, threads>>> (ccs, (num_phi + num_branches) * num_roots, kappa_tot, shear, theta_e, stars, kappa_star, tree[0],
			rectangular, corner, approx, taylor_smooth, caustics);
		if (cuda_error("find_caustics_kernel", true, __FILE__, __LINE__)) return false;
		t_caustics = stopwatch.stop();
		std::cout << "Done finding caustic positions. Elapsed time: " << t_caustics << " seconds.\n\n";

		return true;
	}

	bool write_files(bool verbose)
	{
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
			return false;
		}
		outfile << "kappa_tot " << kappa_tot << "\n";
		outfile << "shear " << shear << "\n";
		outfile << "mu_ave " << mu_ave << "\n";
		outfile << "smooth_fraction " << (1 - kappa_star / kappa_tot) << "\n";
		outfile << "kappa_star " << kappa_star << "\n";
		if (starfile == "")
		{
			outfile << "kappa_star_actual " << kappa_star_actual << "\n";
		}
		outfile << "theta_e " << theta_e << "\n";
		outfile << "random_seed " << random_seed << "\n";
		if (starfile == "")
		{
			outfile << "mass_function " << mass_function_str << "\n";
			if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
			{
				outfile << "m_solar " << m_solar << "\n";
			}
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "mean_mass " << mean_mass << "\n";
			outfile << "mean_mass2 " << mean_mass2 << "\n";
		}
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		outfile << "num_stars " << num_stars << "\n";
		if (rectangular)
		{
			outfile << "corner_x1 " << corner.re << "\n";
			outfile << "corner_x2 " << corner.im << "\n";
			if (approx)
			{
				outfile << "taylor_smooth " << taylor_smooth << "\n";
			}
		}
		else
		{
			outfile << "rad " << corner.abs() << "\n";
		}
		outfile << "num_roots " << num_roots << "\n";
		outfile << "num_phi " << num_phi << "\n";
		outfile << "num_branches " << num_branches << "\n";
		outfile << "max_error_1/mu " << max_error << "\n";
		outfile << "t_init_roots " << t_init_roots << "\n";
		outfile << "t_ccs " << t_ccs << "\n";
		outfile << "t_caustics " << t_caustics << "\n";
		outfile.close();
		std::cout << "Done writing parameter info to file " << fname << "\n\n";


		std::cout << "Writing star info...\n";
		fname = outfile_prefix + "ccf_stars" + outfile_type;
		if (!write_star_file<T>(num_stars, rectangular, corner, theta_e, stars, fname))
		{
			std::cerr << "Error. Unable to write star info to file " << fname << "\n";
			return false;
		}
		std::cout << "Done writing star info to file " << fname << "\n\n";


		/******************************************************************************
		write critical curve positions
		******************************************************************************/
		std::cout << "Writing critical curve positions...\n";
		fname = outfile_prefix + "ccf_ccs" + outfile_type;
		if (!write_array<Complex<T>>(ccs, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write ccs info to file " << fname << "\n";
			return false;
		}
		std::cout << "Done writing critical curve positions to file " << fname << "\n\n";


		/******************************************************************************
		write caustic positions
		******************************************************************************/
		std::cout << "Writing caustic positions...\n";
		fname = outfile_prefix + "ccf_caustics" + outfile_type;
		if (!write_array<Complex<T>>(caustics, num_roots * num_branches, num_phi / num_branches + 1, fname))
		{
			std::cerr << "Error. Unable to write caustic info to file " << fname << "\n";
			return false;
		}
		std::cout << "Done writing caustic positions to file " << fname << "\n\n";

		return true;
	}


public:

	bool run(bool verbose)
	{
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!populate_star_array(verbose)) return false;
		if (!create_tree(verbose)) return false;
		if (!find_initial_roots(verbose)) return false;
		if (!find_ccs_caustics(verbose)) return false;

		return true;
	}

	bool save(bool verbose)
	{
		if (!write_files(verbose)) return false;

		return true;
	}

};

