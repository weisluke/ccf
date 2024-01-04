﻿#pragma once

#include "alpha_local.cuh"
#include "alpha_smooth.cuh"
#include "alpha_star.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************************************************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x > 0, 0 if x <= 0
******************************************************************************/
template <typename T>
__device__ T heaviside(T x)
{
	if (x > 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within the rectangle defined by corner, 0 if it is on the
		border or outside
******************************************************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re < z.re && z.re < corner.re && -corner.im < z.im && z.im < corner.im)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
calculate the deflection angle due to smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate

\return alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> smooth_deflection(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> alpha_smooth;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> s1;
			Complex<T> s2;
			Complex<T> s3;
			Complex<T> s4;

			for (int i = taylor_smooth; i >= 1; i--)
			{
				s1 *= i;
				s1 += 1;
				s1 /= i;

				s2 *= i;
				s2 += 1;
				s2 /= i;

				s3 *= i;
				s3 += 1;
				s3 /= i;

				s4 *= i;
				s4 += 1;
				s4 /= i;

				s1 *= (z.conj() / corner);
				s2 *= (z.conj() / corner.conj());
				s3 *= (z.conj() / -corner);
				s4 *= (z.conj() / -corner.conj());
			}

			alpha_smooth = ((corner - z.conj()) * (corner.log() - s1) - (corner.conj() - z.conj()) * (corner.conj().log() - s2)
				+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4));
			alpha_smooth *= Complex<T>(0, -kappastar / PI);
			alpha_smooth -= kappastar * 2 * (corner.re + z.re);
		}
		else
		{
			Complex<T> c1 = corner - z.conj();
			Complex<T> c2 = corner.conj() - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			alpha_smooth = (c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log());
			alpha_smooth *= Complex<T>(0, -kappastar / PI);
			alpha_smooth -= kappastar * 2 * (corner.re + z.re) * boxcar(z, corner);
			alpha_smooth -= kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);
		}
	}
	else
	{
		alpha_smooth = -kappastar * z;
	}

	return alpha_smooth;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to z

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not

\return d_alpha_smooth_d_z
******************************************************************************/
template <typename T>
__device__ T d_smooth_deflection_d_z(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx)
{
	T d_alpha_smooth_d_z = -kappastar;

	if (rectangular && !approx)
	{
		d_alpha_smooth_d_z *= boxcar(z, corner);
	}

	return d_alpha_smooth_d_z;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to zbar

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate

\return d_alpha_smooth_d_zbar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_smooth_deflection_d_zbar(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> d_alpha_smooth_d_zbar;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r1 = z.conj() / corner;
			Complex<T> r2 = z.conj() / corner.conj();

			Complex<T> s1;
			Complex<T> s2;

			for (int i = (taylor_smooth % 2 == 0 ? taylor_smooth : taylor_smooth - 1); i >= 2; i -= 2)
			{
				s1 *= i;
				s1 += 1;
				s1 /= i;

				s2 *= i;
				s2 += 1;
				s2 /= i;

				s1 *= (r1 * r1);
				s2 *= (r2 * r2);
			}
			d_alpha_smooth_d_zbar += s1 - s2;
			d_alpha_smooth_d_zbar *= 2;

			if (taylor_smooth % 2 == 0)
			{
				d_alpha_smooth_d_zbar += r1.pow(taylor_smooth) * 2;
				d_alpha_smooth_d_zbar -= r2.pow(taylor_smooth) * 2;
			}

			d_alpha_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_alpha_smooth_d_zbar += kappastar - 4 * kappastar * corner.arg() / PI;
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d_alpha_smooth_d_zbar = (c1.log() - c2.log() - c3.log() + c4.log());
			d_alpha_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_alpha_smooth_d_zbar -= kappastar * boxcar(z, corner);
		}
	}

	return d_alpha_smooth_d_zbar;
}

/******************************************************************************
calculate the second derivative of the deflection angle due to smooth matter
with respect to zbar^2

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate

\return d2_alpha_smooth_d_zbar2
******************************************************************************/
template <typename T>
__device__ Complex<T> d2_smooth_deflection_d_zbar2(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> d2_alpha_smooth_d_zbar2;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r1 = z.conj() / corner;
			Complex<T> r2 = z.conj() / corner.conj();

			Complex<T> s1;
			Complex<T> s2;

			for (int i = (taylor_smooth % 2 == 0 ? taylor_smooth : taylor_smooth - 1); i >= 2; i -= 2)
			{
				s1 += 1;
				s2 += 1;

				s1 *= (r1 * r1);
				s2 *= (r2 * r2);
			}
			d2_alpha_smooth_d_zbar2 += s1 - s2;
			d2_alpha_smooth_d_zbar2 /= z.conj();
			d2_alpha_smooth_d_zbar2 *= 2;

			if (taylor_smooth % 2 == 0)
			{
				d2_alpha_smooth_d_zbar2 += taylor_smooth / corner * r1.pow(taylor_smooth - 1) * 2;
				d2_alpha_smooth_d_zbar2 -= taylor_smooth / corner.conj() * r2.pow(taylor_smooth - 1) * 2;
			}

			d2_alpha_smooth_d_zbar2 *= Complex<T>(0, -kappastar / PI);
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d2_alpha_smooth_d_zbar2 = (-1 / c1 + 1 / c2 + 1 / c3 - 1 / c4);
			d2_alpha_smooth_d_zbar2 *= Complex<T>(0, -kappastar / PI);
		}
	}

	return d2_alpha_smooth_d_zbar2;
}

/******************************************************************************
lens equation from image plane to source plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return w = (1 - kappa) * z + gamma * z_bar 
            - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> a_star = alpha_star<T>(z, theta, stars, node);
	Complex<T> a_local = alpha_local<T>(z, theta, node);
	Complex<T> alpha_smooth = smooth_deflection<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	(1 - kappa) * z + gamma * z_bar - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return (1 - kappa) * z + gamma * z.conj() - a_star - a_local - alpha_smooth;
}

/******************************************************************************
parametric critical curve equation for a star field
we seek the values of z that make this equation equal to 0 for a given phi

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param phi -- value of the variable parametrizing z

\return gamma - (d_alpha_star / d_zbar)_bar - (d_alpha_smooth / d_zbar)_bar
		- (1 - kappa - d_alpha_smooth / dz) * e^(-i * phi)
******************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, T phi)
{
	Complex<T> d_a_star_d_zbar = d_alpha_star_d_zbar(z, theta, stars, node);
	Complex<T> d_a_local_d_zbar = d_alpha_local_d_zbar(z, theta, node);
	T d_alpha_smooth_d_z = d_smooth_deflection_d_z(z, kappastar, rectangular, corner, approx);
	Complex<T> d_alpha_smooth_d_zbar = d_smooth_deflection_d_zbar(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	gamma - (d_alpha_star / d_zbar)_bar - (d_alpha_local / d_zbar)_bar
	- (d_alpha_smooth / d_zbar)_bar
	- (1 - kappa - d_alpha_smooth / d_z) * e^(-i * phi)
	******************************************************************************/
	return gamma - d_a_star_d_zbar.conj() - d_a_local_d_zbar.conj()
		- d_alpha_smooth_d_zbar.conj()
		- (1 - kappa - d_alpha_smooth_d_z) * Complex<T>(cos(phi), -sin(phi));
}

/******************************************************************************
derivative of the parametric critical curve equation with respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate

\return -2 * theta^2 * sum(m_i / (z - z_i)^3)
		- (d^2alpha_smooth / dz_bar^2)_bar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d2_a_star_d_zbar2 = d2_alpha_star_d_zbar2(z, theta, stars, node);
	Complex<T> d2_a_local_d_zbar2 = d2_alpha_local_d_zbar2(z, theta, node);
	Complex<T> d2_alpha_smooth_d_zbar2 = d2_smooth_deflection_d_zbar2(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	-(d2_alpha_star / d_zbar2)_bar - (d2_alpha_local / d_zbar2)_bar
	- (d2_alpha_smooth / d_zbar2)_bar
	******************************************************************************/
	return -d2_a_star_d_zbar2.conj() - d2_a_local_d_zbar2.conj() - d2_alpha_smooth_d_zbar2.conj();
}

/******************************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots

\param k -- index of z within the roots array
			0 <= k < nroots
\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param root -- pointer to root node
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
******************************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, T phi, Complex<T>* roots, int nroots)
{
	TreeNode<T>* node = treenode::get_nearest_node(z, root);

	Complex<T> f0 = parametric_critical_curve(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, phi);
	T d_alpha_smooth_d_z = d_smooth_deflection_d_z(z, kappastar, rectangular, corner, approx);

	/******************************************************************************
	if 1/mu < 10^-9, return same position
	the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired
	******************************************************************************/
	if (fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa - d_alpha_smooth_d_z))) < 0.000000001 &&
		fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa - d_alpha_smooth_d_z))) < 0.000000001)
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	contribution due to distance between root and stars
	******************************************************************************/
	Complex<T> starsum;
	for (int i = 0; i < root->numstars; i++)
	{
		starsum += 1 / (z - stars[root->stars + i].position);
	}
	starsum *= 2;

	/******************************************************************************
	contribution due to distance between root and other roots
	******************************************************************************/
	Complex<T> rootsum;
	for (int i = 0; i < nroots; i++)
	{
		if (i != k)
		{
			rootsum += 1 / (z - roots[i]);
		}
	}

	Complex<T> result = f1 / f0 + starsum - rootsum;
	return z - 1 / result;
}

/******************************************************************************
take the list of all roots z, the given value of j out of nphi steps, and the
number of branches, and set the initial roots for step j equal to the final
roots of step j-1
reset values for whether roots have all been found to sufficient accuracy to
false

\param z -- pointer to array of root positions
\param nroots -- number of roots
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots have been
			  found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void prepare_roots_kernel(Complex<T>* z, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			for (int a = x_index; a < nroots; a += x_stride)
			{
				int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;

				if (b == 0)
				{

					z[center + j * nroots + a] = z[center + (j - 1) * nroots + a];
					fin[c * 2 * nroots + a] = false;
				}
				else
				{
					z[center - j * nroots + a] = z[center - (j - 1) * nroots + a];
					fin[c * 2 * nroots + a + nroots] = false;
				}
			}
		}
	}
}

/******************************************************************************
find new critical curve roots for a star field

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param root -- pointer to root node
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots have been
			  found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	Complex<T> result;
	T norm;
	int sgn;

	T PI = static_cast<T>(3.1415926535898);
	T dphi = 2 * PI / nphi * j;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			T phi0 = PI / nbranches + c * 2 * PI / nbranches;

			for (int a = x_index; a < nroots; a += x_stride)
			{
				/******************************************************************************
				we use the following variable to determine whether we are on the positive or
				negative side of phi0, as we are simultaneously growing 2 sets of roots after
				having stepped away from the middle by j out of nphi steps
				******************************************************************************/
				sgn = (b == 0 ? -1 : 1);

				/******************************************************************************
				if root has not already been calculated to desired precision
				we are calculating nbranches * 2 * nroots roots in parallel, so
				" c * 2 * nroots " indicates what branch we are in,
				" b * nroots " indicates whether we are on the positive or negative side, and
				" a " indicates the particular root position
				******************************************************************************/
				if (!fin[c * 2 * nroots + b * nroots + a])
				{
					/******************************************************************************
					calculate new root
					center of the roots array (ie the index of phi0) for all branches is
					( nphi / (2 * nbranches) + c  * nphi / nbranches + c) * nroots
					for the particular value of phi here (i.e., phi0 +/- dphi), roots start
					at +/- j*nroots of that center
					a is then added to get the final index of this particular root
					******************************************************************************/

					int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;
					result = find_critical_curve_root(a, roots[center + sgn * j * nroots + a], kappa, gamma, theta, stars, kappastar, root, rectangular, corner, approx, taylor_smooth, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

					/******************************************************************************
					distance between old root and new root in units of theta_e
					******************************************************************************/
					norm = (result - roots[center + sgn * j * nroots + a]).abs() / theta;

					/******************************************************************************
					compare position to previous value, if less than desired precision of 10^-9,
					set fin[root] to true
					******************************************************************************/
					if (norm < 0.000000001)
					{
						fin[c * 2 * nroots + b * nroots + a] = true;
					}
					roots[center + sgn * j * nroots + a] = result;
				}
			}
		}
	}
}

/******************************************************************************
find maximum error in critical curve roots for a star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param root -- pointer to root node
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array is of size nbranches * 2 * nroots
******************************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, int j, int nphi, int nbranches, T* errs)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	int z_index = blockIdx.z * blockDim.z + threadIdx.z;
	int z_stride = blockDim.z * gridDim.z;

	int sgn;

	T PI = static_cast<T>(3.1415926535898);
	T dphi = 2 * PI / nphi * j;

	for (int c = z_index; c < nbranches; c += z_stride)
	{
		for (int b = y_index; b < 2; b += y_stride)
		{
			T phi0 = PI / nbranches + c * 2 * PI / nbranches;

			for (int a = x_index; a < nroots; a += x_stride)
			{
				sgn = (b == 0 ? -1 : 1);

				int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;

				TreeNode<T>* node = treenode::get_nearest_node(z[center + sgn * j * nroots + a], root);

				/******************************************************************************
				the value of 1/mu depends on the value of f0
				this calculation ensures that the maximum possible value of 1/mu is given
				******************************************************************************/
				Complex<T> f0 = parametric_critical_curve(z[center + sgn * j * nroots + a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, phi0 + sgn * dphi);
				T d_alpha_smooth_d_z = d_smooth_deflection_d_z(z[center + sgn * j * nroots + a], kappastar, rectangular, corner, approx);

				T e1 = fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa - d_alpha_smooth_d_z)));
				T e2 = fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa - d_alpha_smooth_d_z)));

				/******************************************************************************
				return maximum possible error in 1/mu at root position
				******************************************************************************/
				errs[center + sgn * j * nroots + a] = fmax(e1, e2);
			}
		}
	}
}

/******************************************************************************
determine whether errors have nan values

\param errs -- pointer to array of errors
\param nerrs -- number of errors in array
\param hasnan -- pointer to int (bool) of whether array has nan values or not
******************************************************************************/
template <typename T>
__global__ void has_nan_err_kernel(T* errs, int nerrs, int* hasnan)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nerrs; a += x_stride)
	{
		if (!isfinite(errs[a]))
		{
			atomicExch(hasnan, 1);
		}
	}
}

/******************************************************************************
set values in error array equal to max of element in first half and
corresponding partner in second half

\param errs -- pointer to array of errors
\param nerrs -- half the number of errors in array
******************************************************************************/
template <typename T>
__global__ void max_err_kernel(T* errs, int nerrs)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nerrs; a += x_stride)
	{
		errs[a] = fmax(errs[a], errs[a + nerrs]);
	}
}

/******************************************************************************
find caustics from critical curves for a star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param root -- pointer to root node
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param w -- pointer to array of caustic positions
******************************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		TreeNode<T>* node = treenode::get_nearest_node(z[a], root);

		/******************************************************************************
		map image plane positions to source plane positions
		******************************************************************************/
		w[a] = complex_image_to_source(z[a], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
	}
}

/******************************************************************************
transpose array

\param z1 -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param z2 -- pointer to transposed array of values
******************************************************************************/
template <typename T>
__global__ void transpose_array_kernel(Complex<T>* z1, int nrows, int ncols, Complex<T>* z2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nrows * ncols; a += x_stride)
	{
		int col = a % ncols;
		int row = (a - col) / ncols;

		z2[col * nrows + row] = z1[a];
	}
}

/******************************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.open(fname, std::ios_base::binary);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	outfile.write((char*)(&nrows), sizeof(int));
	outfile.write((char*)(&ncols), sizeof(int));
	outfile.write((char*)vals, nrows * ncols * sizeof(T));
	outfile.close();

	return true;
}