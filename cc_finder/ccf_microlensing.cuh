﻿#pragma once

#include "complex.cuh"
#include "star.cuh"


/********************************************************************************
lens equation

\param z -- image plane position
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star

\return w = (1-kappa_smooth)*z + shear*z_bar - theta^2 * sum( m_i / (z-z_i)_bar )
********************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*(1-kappa_smooth)*z+gamma*z_bar-starsum_bar*/
	return z * (1.0 - kappasmooth) + gamma * z.conj() - starsum.conj();
}

/*******************************************************************************
value of the parametric critical curve equation
we seek the values of z that make this equation equal to 0 for a given phi

\param z -- image plane position
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param phi -- value of the variable parametrizing z

\return shear + theta^2 * sum( m_i / (z-z_i)^2 ) - (1-kappa_smooth) * e^(-i*phi)
*******************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, T phi)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^2*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*gamma-(1-kappa_smooth)*e^(-i*phi)+starsum*/
	return starsum + gamma - (1.0 - kappasmooth) * Complex<T>(cos(phi), -sin(phi));
}

/*********************************************************************
derivative of the parametric critical curve equation with respect to z

\param z -- image plane position
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star

\return -2 * theta^2 * sum( m_i / (z-z_i)^3 )

*********************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^3*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position) * (z - stars[i].position));
	}

	/*-2*theta_e^2*starsum*/
	return starsum * -2.0 * theta * theta;
}

/************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots
k is the index of the particular root (0 <= k < nroots)

\param k -- index of z within the roots array
\param z -- image plane position
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, T phi, Complex<T>* roots, int nroots)
{
	Complex<T> f0 = parametric_critical_curve<T>(z, stars, nstars, kappasmooth, gamma, theta, phi);

	/*if 1/mu < 10^-9, return same position. the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired*/
	if (fabs(f0.abs() * (f0.abs() + 2.0 * (1.0 - kappasmooth))) < 0.000000001 &&
		fabs(f0.abs() * (f0.abs() - 2.0 * (1.0 - kappasmooth))) < 0.000000001)
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz<T>(z, stars, nstars, kappasmooth, gamma, theta);

	/*contribution due to distance between root and stars*/
	Complex<T> starsum;
	for (int i = 0; i < nstars; i++)
	{
		starsum += 2.0 / (z - stars[i].position);
	}

	/*contribution due to distance between root and other roots*/
	Complex<T> rootsum;
	for (int i = 0; i < nroots; i++)
	{
		if (i != k)
		{
			rootsum += 1.0 / (z - roots[i]);
		}
	}

	Complex<T> result = f1 / f0 + starsum - rootsum;
	return z - 1.0 / result;
}

/*****************************************************************
take the list of all roots z, the given value of j out of nphi
steps, and the number of branches, and set the initial roots for
step j equal to the final roots of step j-1
reset values for whether roots have all been found to
sufficient accuracy to false

\param z -- pointer to array of root positions
\param nroots -- number of roots
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array should be of size nbranches * 2 * nroots
*****************************************************************/
template <typename T>
__global__ void prepare_roots_kernel(Complex<T>* z, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int k = y_index; k < nbranches; k += y_stride)
	{
		for (int i = x_index; i < nroots; i += x_stride)
		{
			int center = (nphi / (2 * nbranches) + k * nphi / nbranches) * nroots;

			z[center + j * nroots + i] = z[center + (j - 1) * nroots + i];
			z[center - j * nroots + i] = z[center - (j - 1) * nroots + i];
			fin[k * 2 * nroots + i] = false;
			fin[k * 2 * nroots + i + nroots] = false;
		}
	}
}

/*****************************************************************
find new critical curve roots

\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array should be of size nbranches * 2 * nroots
*****************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	Complex<T> result;
	T norm;
	int sgn;
	int ipos;

	double PI = 3.1415926535898;
	T dphi = 2.0 * PI / nphi * j;

	for (int k = y_index; k < nbranches; k += y_stride)
	{
		T phi0 = PI / nbranches + k * 2.0 * PI / nbranches;

		for (int i = x_index; i < 2 * nroots; i += x_stride)
		{
			/*we use the following variables to determine whether we are on the positive
			or negative side of phi0, as we are simultaneously growing 2 sets of roots
			after having stepped away from the middle by j out of nphi steps
			sgn determines the side of phi0, and ipos determines the actual index of
			the root that we are recalculating*/
			sgn = (i < nroots ? -1 : 1);
			ipos = i % nroots;

			/*if root has not already been calculated to desired precision
			we are calculating nbranches * 2 * nroots roots in parallel, so
			k * 2 * nroots indicates what branch we are in, with " + i " then
			indicating the particular root position*/
			if (!fin[k * 2 * nroots + i])
			{
				/*calculate new root
				center of the roots array (ie the index of phi0) for all branches is
				( nphi / (2 * nbranches) + k  * nphi / nbranches ) * nroots
				for the particular value of phi here (i.e., phi0 +/- dphi),
				roots start at +/- j*nroots of that center
				ipos is then added to get the final index of this particular root*/

				int center = (nphi / (2 * nbranches) + k * nphi / nbranches) * nroots;
				result = find_critical_curve_root<T>(ipos, roots[center + sgn * j * nroots + ipos], stars, nstars, kappasmooth, gamma, theta, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

				/*distance between old root and new root in units of theta_e*/
				norm = (result - roots[center + sgn * j * nroots + ipos]).abs() / theta;

				/*compare position to previous value, if less than desired precision of 10^-9, set fin[root] to true*/
				if (norm < 0.000000001)
				{
					fin[k * 2 * nroots + i] = true;
				}
				roots[center + sgn * j * nroots + ipos] = result;
			}
		}
	}
}

/**************************************************************
find maximum error in critical curve roots

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array should be of size nbranches * 2 * nroots
**************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, int j, int nphi, int nbranches, T* errs)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int sgn;
	int ipos;
	
	double PI = 3.1415926535898;
	T dphi = 2.0 * PI / nphi * j;

	for (int k = y_index; k < nbranches; k += y_stride)
	{
		T phi0 = PI / nbranches + k * 2.0 * PI / nbranches;

		for (int i = x_index; i < 2 * nroots; i += x_stride)
		{
			sgn = (i < nroots ? -1 : 1);
			ipos = i % nroots;

			int center = (nphi / (2 * nbranches) + k * nphi / nbranches) * nroots;

			/*the value of 1/mu depends on the value of f0
			this calculation ensures that the maximum possible value of 1/mu is given*/
			Complex<T> f0 = parametric_critical_curve<T>(z[center + sgn * j * nroots + ipos], stars, nstars, kappasmooth, gamma, theta, phi0 + sgn * dphi);

			T e1 = fabs(f0.abs() * (f0.abs() + 2.0 * (1.0 - kappasmooth)));
			T e2 = fabs(f0.abs() * (f0.abs() - 2.0 * (1.0 - kappasmooth)));

			/*return maximum possible error in 1/mu at root position*/
			errs[center + sgn * j * nroots + ipos] = fmax(e1, e2);
		}
	}
}

/******************************************************
determine whether errors have nan values

\param errs -- pointer to array of errors
\param nerrs -- number of errors in array
\param hasnan -- pointer to int (bool) of whether array
				 has nan values or not
******************************************************/
template <typename T>
__global__ void has_nan_err_kernel(T* errs, int nerrs, int* hasnan)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nerrs; i += x_stride)
	{
		if (!isfinite(errs[i]))
		{
			atomicExch(hasnan, 1);
		}
	}
}

/******************************************************
set values in error array equal to max of element in
first half and corresponding partner in second half

\param errs -- pointer to array of errors
\param nerrs -- half the number of errors in array
******************************************************/
template <typename T>
__global__ void max_err_kernel(T* errs, int nerrs)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nerrs; i += x_stride)
	{
		errs[i] = fmax(errs[i], errs[i + nerrs]);
	}
}

/**************************************************************
find caustics from critical curves

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param w -- pointer to array of caustic positions
**************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nroots; i += x_stride)
	{
		/*map image plane positions to source plane positions*/
		w[i] = complex_image_to_source<T>(z[i], stars, nstars, kappasmooth, gamma, theta);
	}
}

/**************************************************************
transpose array

\param z1 -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param z2 -- pointer to transposed array of values
**************************************************************/
template <typename T>
__global__ void transpose_array_kernel(Complex<T>* z1, int nrows, int ncols, Complex<T>* z2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nrows * ncols; i += x_stride)
	{
		int col = i % ncols;
		int row = (i - col) / ncols;

		z2[col * nrows + row] = z1[i];
	}
}

