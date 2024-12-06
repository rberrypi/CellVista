#include "stdafx.h"
#include "acquisition_framework.h"
#include "compute_engine.h"
#include <numeric>

template <typename Iterator, typename Cont>
bool is_last(Iterator iterator, const Cont& cont)
{
	return (iterator != cont.end()) && (next(iterator) == cont.end());
}

float acquisition_framework::process_focus_list(const std::vector<auto_focus_info>& values)
{
	//Z, Focus
	const auto comparator = [](const auto_focus_info& a, const auto_focus_info& b)
	{
		return a.metric < b.metric;
	};
	const auto max_it = std::max_element(values.begin(), values.end(), comparator);

	if (max_it == values.begin() || is_last(max_it, values))
	{
		const auto predicate = [values](const auto_focus_info& x)
		{
			return x.metric == values[0].metric;
		};
		const auto some_kind_of_failure = std::all_of(values.begin(), values.end(), predicate);
		const auto op = [](const float current_sum, const auto_focus_info& b)
		{
			return current_sum + b.z;
		};
		const auto middle = std::accumulate(values.begin(), values.end(), 0.0f, op) / values.size();
		return some_kind_of_failure ? middle : max_it->z;
	}
	const auto ai = std::prev(max_it);
	const auto bi = max_it;
	const auto ci = std::next(max_it);
	return parabolic_peak(ai->z, ai->metric, bi->z, bi->metric, ci->z, ci->metric);
}

float acquisition_framework::parabolic_peak(const float i_a, const float v_a, const float i_b, const float v_b, const float i_c, const float v_c)
{
	//http://www.dsprelated.com/dspbooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html
	/*
	return ((iA<iB)&&(iB<iC)) ? iB-0.5*(pow(iB-iA,2)*(vB-vC)-pow(iB-iC,2)*(vB-vA))/
	((iB-iA)*(vB-vC)-(iB-iC)*(vB-vA)) : vC;
	*/
	const auto val = i_b - 0.5f * (pow(i_b - i_a, 2) * (v_b - v_c) - pow(i_b - i_c, 2) * (v_b - v_a)) / ((i_b - i_a) * (v_b - v_c) - (i_b - i_c) * (v_b - v_a));
	return std::isfinite(val) ? val : i_b;
}
