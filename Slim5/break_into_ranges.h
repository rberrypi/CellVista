#pragma once
#ifndef BREAK_INTO_RANGES_HPP
#define BREAK_INTO_RANGES_HPP

template<class T, class _Pr>
auto break_into_ranges(const std::vector<T>& values, _Pr& predicate)
{
	//todo, input should also be an iterator?
	std::vector<std::pair<typename std::vector<T>::const_iterator, typename std::vector<T>::const_iterator>>  ranges;
	for (size_t i = 0; i < values.size();)
	{
		auto start = values.begin() + i;
		auto first = *start;
		auto j = i;
		while (j < values.size())
		{
			//if (first != values[j])
			if (!predicate(first, values[j]))
			{
				break;
			}
			j++;
		}
		auto past_end = values.begin() + j;
		i = j;
		ranges.push_back(std::make_pair(start, past_end));
	}
	return ranges;
}

#endif