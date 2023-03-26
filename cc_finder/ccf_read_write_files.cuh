#pragma once

#include "complex.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/**************************************************************
write array of complex values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .bin file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_complex_array(Complex<T>* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	std::ofstream outfile;

	if (fpath.extension() == ".bin")
	{
		outfile.open(fname, std::ios_base::binary);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		outfile.write((char*)(&nrows), sizeof(int));
		outfile.write((char*)(&ncols), sizeof(int));
		outfile.write((char*)vals, nrows * ncols * sizeof(Complex<T>));
		outfile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	return true;
}

/**************************************************************
write array of real part of complex values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .txt file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_re_array(Complex<T>* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	std::ofstream outfile;
	outfile.precision(9);

	if (fpath.extension() == ".txt")
	{
		outfile.open(fname);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				outfile << vals[i * ncols + j].re << " ";
			}
			outfile << "\n";
		}
		outfile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	return true;
}

/**************************************************************
write array of imaginary part of complex values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the .txt file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_im_array(Complex<T>* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	std::ofstream outfile;
	outfile.precision(9);

	if (fpath.extension() == ".txt")
	{
		outfile.open(fname);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				outfile << vals[i * ncols + j].im << " ";
			}
			outfile << "\n";
		}
		outfile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	return true;
}

