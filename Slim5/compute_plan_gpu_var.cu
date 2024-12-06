//#include "stdafx.h"
#include "Thrust_GPU_Var.h"
//https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu


#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
//#include <thrust/extrema.h>
//#include <cmath>
//#include <limits>

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other 
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
	T n;
	//T min;
	//T max;
	T mean;
	T m2;
	//T M3;
	//T M4;

	// initialize to the identity element
	void initialize()
	{
		n = mean = m2 = 0;// = M3 = M4 = 0;
		//min = std::numeric_limits<T>::max();
		//max = std::numeric_limits<T>::min();
	}

	T variance() { return m2 / (n - 1); }
	T variance_n() { return m2 / n; }
	//T skewness()   { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
	//T kurtosis()   { return n * M4 / (M2 * M2); }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
	__host__ __device__
		summary_stats_data<T> operator()(const T& x) const
	{
		summary_stats_data<T> result;
		result.n = 1;
		//result.min = x;
		//result.max = x;
		result.mean = x;
		result.m2 = 0;
		//result.M3 = 0;
		//result.M4 = 0;

		return result;
	}
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op
	: public thrust::binary_function<const summary_stats_data<T>&,
	const summary_stats_data<T>&,
	summary_stats_data<T> >
{
	__host__ __device__
		summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
	{
		summary_stats_data<T> result;

		// precompute some common subexpressions
		T n = x.n + y.n;
		//T n2 = n  * n;
		//T n3 = n2 * n;

		T delta = y.mean - x.mean;
		T delta2 = delta * delta;
		//T delta3 = delta2 * delta;
		//T delta4 = delta3 * delta;

		//Basic number of samples (n), min, and max
		result.n = n;
		//result.min = thrust::min(x.min, y.min);
		//result.max = thrust::max(x.max, y.max);

		result.mean = x.mean + delta * y.n / n;

		result.m2 = x.m2 + y.m2;
		result.m2 += delta2 * x.n * y.n / n;
		/*
		result.M3 = x.M3 + y.M3;
		result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
		result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;

		result.M4 = x.M4 + y.M4;
		result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
		result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
		result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;
		*/
		return result;
	}
};

float thrust_gpu_var::get_variance(const thrust::device_vector<float>::iterator start, const thrust::device_vector<float>::iterator stop) const
{
	typedef float T;
	summary_stats_unary_op<T>  unary_op;
	summary_stats_binary_op<T> binary_op;
	summary_stats_data<T>      init;
	init.initialize();
	auto result = thrust::transform_reduce(start, stop, unary_op, init, binary_op);
	auto val = result.variance();
	return val;
}
