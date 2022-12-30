#pragma once

#include "complex.cuh"
#include "star.cuh"


/******************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x > 0, 0 if x <= 0
******************************/
template <typename T>
__device__ T heaviside(T x)
{
	if (x > 0)
	{
		return static_cast<T>(1);
	}
	else
	{
		return static_cast<T>(0);
	}
}

/************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within the rectangle
		defined by corner, 0 if it is on the
		border or outside
************************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re < z.re && z.re < corner.re && -corner.im < z.im && z.im < corner.im)
	{
		return static_cast<T>(1);
	}
	else
	{
		return static_cast<T>(0);
	}
}

/********************************************************************
lens equation for a rectangular star field

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses

\return w = (1 - kappa) * z + gamma * z_bar
            - theta^2 * sum(m_i / (z - z_i)_bar) - alpha_smooth
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> c1 = corner.conj() - z.conj();
	Complex<T> c2 = corner - z.conj();
	Complex<T> c3 = -corner - z.conj();
	Complex<T> c4 = -corner.conj() - z.conj();

	Complex<T> alpha_smooth = Complex<T>(0, -kappastar / PI) * (-c1 * c1.log() + c2 * c2.log() + c3 * c3.log() - c4 * c4.log())
		- kappastar * 2 * (corner.re + z.re) * boxcar(z, corner)
		- kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);

	/*(1-kappa)*z+gamma*z_bar-starsum_bar-alpha_smooth*/
	return (1 - kappa) * z + gamma * z.conj() - starsum.conj() - alpha_smooth;
}

/********************************************************************
approximate lens equation for a rectangular star field

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses

\return w = (1 - kappa) * z + gamma * z_bar
			- theta^2 * sum(m_i / (z - z_i)_bar) - alpha_smooth
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> s1;
	Complex<T> s2;
	Complex<T> s3;
	Complex<T> s4;

	for (int i = 1; i <= taylor; i++)
	{
		s1 += (z.conj() / corner.conj()).pow(i) / i;
		s2 += (z.conj() / corner).pow(i) / i;
		s3 += (z.conj() / -corner).pow(i) / i;
		s4 += (z.conj() / -corner.conj()).pow(i) / i;
	}

	Complex<T> alpha_smooth = (-(corner.conj() - z.conj()) * (corner.conj().log() - s1) + (corner - z.conj()) * (corner.log() - s2)
		+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4));
	alpha_smooth *= Complex<T>(0, -kappastar / PI);
	alpha_smooth -= kappastar * (corner + corner.conj() + z + z.conj());

	/*(1-kappa)*z+gamma*z_bar-starsum_bar-alpha_smooth*/
	return (1 - kappa) * z + gamma * z.conj() - starsum.conj() - alpha_smooth;
}

/********************************************************************
lens equation for a circular star field

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses

\return w = (1 - kappa + kappastar) * z + gamma * z_bar
            - theta^2 * sum(m_i / (z - z_i)_bar)
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*(1-(kappa-kappastar))*z+gamma*z_bar-starsum_bar*/
	return (1 - kappa + kappastar) * z + gamma * z.conj() - starsum.conj();
}

/******************************************************************************
parametric critical curve equation for a rectangular star field
we seek the values of z that make this equation equal to 0 for a given phi

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param phi -- value of the variable parametrizing z

\return gamma + theta^2 * sum(m_i / (z - z_i)^2) - (dalpha_smooth / dz_bar)_bar
        - (1 - kappa + kappastar * boxcar(z, corner)) * e^(-i * phi)
******************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, T phi)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^2*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> c1 = corner.conj() - z.conj();
	Complex<T> c2 = corner - z.conj();
	Complex<T> c3 = -corner - z.conj();
	Complex<T> c4 = -corner.conj() - z.conj();

	Complex<T> dalpha_smooth_dz_bar = Complex<T>(0, -kappastar / PI) * (c1.log() - c2.log() - c3.log() + c4.log())
		- kappastar * boxcar(z, corner);

	/*gamma+starsum-(1-kappa+kappastar*boxcar))*e^(-i*phi)*/
	return gamma + starsum - dalpha_smooth_dz_bar.conj() - (1 - kappa + kappastar * boxcar(z, corner)) * Complex<T>(cos(phi), -sin(phi));
}

/******************************************************************************
parametric critical curve equation for a rectangular star field
with approximations
we seek the values of z that make this equation equal to 0 for a given phi

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param phi -- value of the variable parametrizing z

\return gamma + theta^2 * sum(m_i / (z - z_i)^2) - (dalpha_smooth / dz_bar)_bar
		- (1 - kappa + kappastar * boxcar(z, corner)) * e^(-i * phi)
******************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, T phi)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^2*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> r1 = z.conj() / corner;
	Complex<T> r2 = z.conj() / corner.conj();

	Complex<T> dalpha_smooth_dz_bar;

	for (int i = 2; i <= taylor; i += 2)
	{
		dalpha_smooth_dz_bar += (r1.pow(i) - r2.pow(i)) / i;
	}
	dalpha_smooth_dz_bar *= 2;

	if (taylor % 2 == 0)
	{
		dalpha_smooth_dz_bar += r1.pow(taylor) * 2;
		dalpha_smooth_dz_bar -= r2.pow(taylor) * 2;
	}

	dalpha_smooth_dz_bar *= Complex<T>(0, -kappastar / PI);
	dalpha_smooth_dz_bar += kappastar - 4 * kappastar * corner.arg() / PI;

	/*gamma+starsum-(1-kappa+kappastar*boxcar))*e^(-i*phi)*/
	return gamma + starsum - dalpha_smooth_dz_bar.conj() - (1 - kappa + kappastar) * Complex<T>(cos(phi), -sin(phi));
}

/*************************************************************************
parametric critical curve equation for a circular star field
we seek the values of z that make this equation equal to 0 for a given phi

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param phi -- value of the variable parametrizing z

\return theta^2 * sum(m_i / (z - z_i)^2) + gamma
        - (1 - kappa + kappastar) * e^(-i * phi)
*************************************************************************/
template <typename T>
__device__ Complex<T> parametric_critical_curve(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, T phi)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^2*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*gamma+starsum-(1-kappasmooth)*e^(-i*phi)*/
	return gamma + starsum - (1 - kappa + kappastar) * Complex<T>(cos(phi), -sin(phi));
}

/********************************************************************
derivative of the parametric critical curve equation for
a rectangular star field with respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses

\return -2 * theta^2 * sum(m_i / (z - z_i)^3)
        - (d^2alpha_smooth / dz_bar^2)_bar
********************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^3*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> c1 = corner.conj() - z.conj();
	Complex<T> c2 = corner - z.conj();
	Complex<T> c3 = -corner - z.conj();
	Complex<T> c4 = -corner.conj() - z.conj();

	Complex<T> d2alpha_smooth_dz_bar2 = Complex<T>(0, -kappastar / PI) * (-1 / c1 + 1 / c2 + 1 / c3 - 1 / c4);

	/*-2*starsum - (d2alpha_dzbar2)bar*/
	return -2 * starsum - d2alpha_smooth_dz_bar2.conj();
}

/********************************************************************
derivative of the parametric critical curve equation for
a rectangular star field with approximations with respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth

\return -2 * theta^2 * sum(m_i / (z - z_i)^3)
        - (d^2alpha_smooth / dz_bar^2)_bar
********************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^3*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> r1 = z.conj() / corner;
	Complex<T> r2 = z.conj() / corner.conj();

	Complex<T> d2alpha_smooth_dz_bar2;

	for (int i = 2; i <= taylor; i += 2)
	{
		d2alpha_smooth_dz_bar2 += (r1.pow(i - 1) / corner - r2.pow(i - 1) / corner.conj());
	}
	d2alpha_smooth_dz_bar2 *= 2;

	if (taylor % 2 == 0)
	{
		d2alpha_smooth_dz_bar2 += taylor / corner * r1.pow(taylor - 1) * 2;
		d2alpha_smooth_dz_bar2 -= taylor / corner.conj() * r2.pow(taylor - 1) * 2;
	}

	d2alpha_smooth_dz_bar2 *= Complex<T>(0, -kappastar / PI);

	/*-2*starsum - (d2alpha_dzbar2)bar*/
	return -2 * starsum - d2alpha_smooth_dz_bar2.conj();
}

/*********************************************************************
derivative of the parametric critical curve equation for
a circular star field with respect to z

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses

\return -2 * theta^2 * sum(m_i / (z - z_i)^3)
*********************************************************************/
template <typename T>
__device__ Complex<T> d_parametric_critical_curve_dz(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)^3*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / ((z - stars[i].position) * (z - stars[i].position) * (z - stars[i].position));
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*-2*starsum*/
	return -2 * starsum;
}

/************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots
for a rectangular star field

\param k -- index of z within the roots array
			0 <= k < nroots
\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, T phi, Complex<T>* roots, int nroots)
{
	Complex<T> f0 = parametric_critical_curve<T>(z, kappa, gamma, theta, stars, nstars, kappastar, corner, phi);

	/*if 1/mu < 10^-9, return same position. the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired*/
	if (fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar * boxcar(z, corner)))) < static_cast<T>(0.000000001) &&
		fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar * boxcar(z, corner)))) < static_cast<T>(0.000000001))
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz<T>(z, kappa, gamma, theta, stars, nstars, kappastar, corner);

	/*contribution due to distance between root and stars*/
	Complex<T> starsum;
	for (int i = 0; i < nstars; i++)
	{
		starsum += 2 / (z - stars[i].position);
	}

	/*contribution due to distance between root and other roots*/
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

/************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots
for a rectangular star field with approximations

\param k -- index of z within the roots array
			0 <= k < nroots
\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, T phi, Complex<T>* roots, int nroots)
{
	Complex<T> f0 = parametric_critical_curve<T>(z, kappa, gamma, theta, stars, nstars, kappastar, corner, taylor, phi);

	/*if 1/mu < 10^-9, return same position. the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired*/
	if (fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar))) < static_cast<T>(0.000000001) &&
		fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar))) < static_cast<T>(0.000000001))
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz<T>(z, kappa, gamma, theta, stars, nstars, kappastar, corner, taylor);

	/*contribution due to distance between root and stars*/
	Complex<T> starsum;
	for (int i = 0; i < nstars; i++)
	{
		starsum += 2 / (z - stars[i].position);
	}

	/*contribution due to distance between root and other roots*/
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

/************************************************************
find an updated approximation for a particular critical curve
root given the current approximation z and all other roots
for a circular star field

\param k -- index of z within the roots array
			0 <= k < nroots
\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param phi -- value of the variable parametrizing z
\param roots -- pointer to array of roots
\param nroots -- number of roots in array

\return z_new -- updated value of the root z
************************************************************/
template <typename T>
__device__ Complex<T> find_critical_curve_root(int k, Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, T phi, Complex<T>* roots, int nroots)
{
	Complex<T> f0 = parametric_critical_curve<T>(z, kappa, gamma, theta, stars, nstars, kappastar, phi);

	/*if 1/mu < 10^-9, return same position. the value of 1/mu depends on the value of f0
	this check ensures that the maximum possible value of 1/mu is less than desired*/
	if (fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar))) < static_cast<T>(0.000000001) &&
		fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar))) < static_cast<T>(0.000000001))
	{
		return z;
	}

	Complex<T> f1 = d_parametric_critical_curve_dz<T>(z, kappa, gamma, theta, stars, nstars, kappastar);

	/*contribution due to distance between root and stars*/
	Complex<T> starsum;
	for (int i = 0; i < nstars; i++)
	{
		starsum += 2 / (z - stars[i].position);
	}

	/*contribution due to distance between root and other roots*/
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

/*****************************************************************
take the list of all roots z, the given value of j out of nphi
steps, and the number of branches, and set the initial roots for
step j equal to the final roots of step j-1
reset values for whether roots have all been found to
sufficient accuracy to false

\param z -- pointer to array of root positions
\param nroots -- number of roots
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
*****************************************************************/
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

/*****************************************************************
find new critical curve roots for a rectangular star field

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
*****************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
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
				/*we use the following variable to determine whether we are on the positive
				or negative side of phi0, as we are simultaneously growing 2 sets of roots
				after having stepped away from the middle by j out of nphi steps*/
				sgn = (b == 0 ? -1 : 1);

				/*if root has not already been calculated to desired precision
				we are calculating nbranches * 2 * nroots roots in parallel, so
				" c * 2 * nroots " indicates what branch we are in, 
				" b * nroots " indicates whether we are on the positive or negative
				side, and " a " indicates the particular root position*/
				if (!fin[c * 2 * nroots + b * nroots + a])
				{
					/*calculate new root
					center of the roots array (ie the index of phi0) for all branches is
					( nphi / (2 * nbranches) + c  * nphi / nbranches + c) * nroots
					for the particular value of phi here (i.e., phi0 +/- dphi),
					roots start at +/- j*nroots of that center
					a is then added to get the final index of this particular root*/

					int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;
					result = find_critical_curve_root<T>(a, roots[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, corner, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

					/*distance between old root and new root in units of theta_e*/
					norm = (result - roots[center + sgn * j * nroots + a]).abs() / theta;

					/*compare position to previous value, if less than desired precision of 10^-9, set fin[root] to true*/
					if (norm < static_cast<T>(0.000000001))
					{
						fin[c * 2 * nroots + b * nroots + a] = true;
					}
					roots[center + sgn * j * nroots + a] = result;
				}
			}
		}
	}
}

/*****************************************************************
find new critical curve roots for a rectangular star field
with approximations

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
*****************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
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
				/*we use the following variable to determine whether we are on the positive
				or negative side of phi0, as we are simultaneously growing 2 sets of roots
				after having stepped away from the middle by j out of nphi steps*/
				sgn = (b == 0 ? -1 : 1);

				/*if root has not already been calculated to desired precision
				we are calculating nbranches * 2 * nroots roots in parallel, so
				" c * 2 * nroots " indicates what branch we are in,
				" b * nroots " indicates whether we are on the positive or negative
				side, and " a " indicates the particular root position*/
				if (!fin[c * 2 * nroots + b * nroots + a])
				{
					/*calculate new root
					center of the roots array (ie the index of phi0) for all branches is
					( nphi / (2 * nbranches) + c  * nphi / nbranches + c) * nroots
					for the particular value of phi here (i.e., phi0 +/- dphi),
					roots start at +/- j*nroots of that center
					a is then added to get the final index of this particular root*/

					int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;
					result = find_critical_curve_root<T>(a, roots[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, corner, taylor, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

					/*distance between old root and new root in units of theta_e*/
					norm = (result - roots[center + sgn * j * nroots + a]).abs() / theta;

					/*compare position to previous value, if less than desired precision of 10^-9, set fin[root] to true*/
					if (norm < static_cast<T>(0.000000001))
					{
						fin[c * 2 * nroots + b * nroots + a] = true;
					}
					roots[center + sgn * j * nroots + a] = result;
				}
			}
		}
	}
}

/*****************************************************************
find new critical curve roots for a circular star field

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param roots -- pointer to array of roots
\param nroots -- number of roots in array
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi]
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param fin -- pointer to array of boolean values for whether roots
			  have been found to sufficient accuracy
			  array is of size nbranches * 2 * nroots
*****************************************************************/
template <typename T>
__global__ void find_critical_curve_roots_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T>* roots, int nroots, int j, int nphi, int nbranches, bool* fin)
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
				/*we use the following variable to determine whether we are on the positive
				or negative side of phi0, as we are simultaneously growing 2 sets of roots
				after having stepped away from the middle by j out of nphi steps*/
				sgn = (b == 0 ? -1 : 1);

				/*if root has not already been calculated to desired precision
				we are calculating nbranches * 2 * nroots roots in parallel, so
				" c * 2 * nroots " indicates what branch we are in,
				" b * nroots " indicates whether we are on the positive or negative
				side, and " a " indicates the particular root position*/
				if (!fin[c * 2 * nroots + b * nroots + a])
				{
					/*calculate new root
					center of the roots array (ie the index of phi0) for all branches is
					( nphi / (2 * nbranches) + c  * nphi / nbranches + c) * nroots
					for the particular value of phi here (i.e., phi0 +/- dphi),
					roots start at +/- j*nroots of that center
					a is then added to get the final index of this particular root*/

					int center = (nphi / (2 * nbranches) + c * nphi / nbranches + c) * nroots;
					result = find_critical_curve_root<T>(a, roots[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, phi0 + sgn * dphi, &(roots[center + sgn * j * nroots]), nroots);

					/*distance between old root and new root in units of theta_e*/
					norm = (result - roots[center + sgn * j * nroots + a]).abs() / theta;

					/*compare position to previous value, if less than desired precision of 10^-9, set fin[root] to true*/
					if (norm < static_cast<T>(0.000000001))
					{
						fin[c * 2 * nroots + b * nroots + a] = true;
					}
					roots[center + sgn * j * nroots + a] = result;
				}
			}
		}
	}
}

/**************************************************************
find maximum error in critical curve roots
for a rectangular star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array is of size nbranches * 2 * nroots
**************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int j, int nphi, int nbranches, T* errs)
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

				/*the value of 1/mu depends on the value of f0
				this calculation ensures that the maximum possible value of 1/mu is given*/
				Complex<T> f0 = parametric_critical_curve<T>(z[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, corner, phi0 + sgn * dphi);

				T e1 = fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar * boxcar(z[center + sgn * j * nroots + a], corner))));
				T e2 = fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar * boxcar(z[center + sgn * j * nroots + a], corner))));

				/*return maximum possible error in 1/mu at root position*/
				errs[center + sgn * j * nroots + a] = fmax(e1, e2);
			}
		}
	}
}

/**************************************************************
find maximum error in critical curve roots
for a rectangular star field with approximations

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array is of size nbranches * 2 * nroots
**************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, int j, int nphi, int nbranches, T* errs)
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

				/*the value of 1/mu depends on the value of f0
				this calculation ensures that the maximum possible value of 1/mu is given*/
				Complex<T> f0 = parametric_critical_curve<T>(z[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, corner, taylor, phi0 + sgn * dphi);

				T e1 = fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar)));
				T e2 = fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar)));

				/*return maximum possible error in 1/mu at root position*/
				errs[center + sgn * j * nroots + a] = fmax(e1, e2);
			}
		}
	}
}

/**************************************************************
find maximum error in critical curve roots
for a circular star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param j -- position in the number of steps used for phi
\param nphi -- total number of steps used for phi in [0, 2*pi
\param nbranches -- total number of branches for phi in [0, 2*pi]
\param errs -- pointer to array of errors
			   array is of size nbranches * 2 * nroots
**************************************************************/
template <typename T>
__global__ void find_errors_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, int j, int nphi, int nbranches, T* errs)
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

				/*the value of 1/mu depends on the value of f0
				this calculation ensures that the maximum possible value of 1/mu is given*/
				Complex<T> f0 = parametric_critical_curve<T>(z[center + sgn * j * nroots + a], kappa, gamma, theta, stars, nstars, kappastar, phi0 + sgn * dphi);

				T e1 = fabs(f0.abs() * (f0.abs() + 2 * (1 - kappa + kappastar)));
				T e2 = fabs(f0.abs() * (f0.abs() - 2 * (1 - kappa + kappastar)));

				/*return maximum possible error in 1/mu at root position*/
				errs[center + sgn * j * nroots + a] = fmax(e1, e2);
			}
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

	for (int a = x_index; a < nerrs; a += x_stride)
	{
		if (!isfinite(errs[a]))
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

	for (int a = x_index; a < nerrs; a += x_stride)
	{
		errs[a] = fmax(errs[a], errs[a + nerrs]);
	}
}

/**************************************************************
find caustics from critical curves for a rectangular star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param w -- pointer to array of caustic positions
**************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		/*map image plane positions to source plane positions*/
		w[a] = complex_image_to_source<T>(z[a], kappa, gamma, theta, stars, nstars, kappastar, corner);
	}
}

/**************************************************************
find caustics from critical curves for a rectangular star field
with approximations

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param w -- pointer to array of caustic positions
**************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		/*map image plane positions to source plane positions*/
		w[a] = complex_image_to_source<T>(z[a], kappa, gamma, theta, stars, nstars, kappastar, corner, taylor);
	}
}

/**************************************************************
find caustics from critical curves for a circular star field

\param z -- pointer to array of roots
\param nroots -- number of roots in array
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param w -- pointer to array of caustic positions
**************************************************************/
template <typename T>
__global__ void find_caustics_kernel(Complex<T>* z, int nroots, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T>* w)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nroots; a += x_stride)
	{
		/*map image plane positions to source plane positions*/
		w[a] = complex_image_to_source<T>(z[a], kappa, gamma, theta, stars, nstars, kappastar);
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

	for (int a = x_index; a < nrows * ncols; a += x_stride)
	{
		int col = a % ncols;
		int row = (a - col) / ncols;

		z2[col * nrows + row] = z1[a];
	}
}

