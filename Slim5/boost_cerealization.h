#pragma once
#ifndef BOOST_CEREALIZATION_H
#define BOOST_CEREALIZATION_H

#include <boost/container/small_vector.hpp>

namespace cereal
{
	template <class Archive, typename T, unsigned N>
	void save(Archive& ar, const boost::container::small_vector<T, N>& items)
	{

		for (auto i = 0; i < items.size(); i++)
		{
			const auto item = items.at(i);
			const auto name = std::to_string(i);
			ar(make_nvp(name, item));
		}

	}
	//
	template <class Archive, typename T, unsigned N>
	void load(Archive& ar, boost::container::small_vector<T, N>& items)
	{

		items.resize(0);
		while (true)
		{
			const auto name_ptr = ar.getNodeName();
			if (!name_ptr)
			{
				break;
			}
			T item;
			ar(item);
			items.push_back(item);
		}

	}
}

namespace cereal
{
	template <class Archive, typename T, unsigned N>
	void save(Archive& ar, const boost::container::static_vector<T, N>& items)
	{

		for (auto i = 0; i < items.size(); i++)
		{
			const auto item = items[i];
			const auto name = std::to_string(i);
			ar(make_nvp(name, item));
		}

	}
	//
	template <class Archive, typename T, unsigned N>
	void load(Archive& ar, boost::container::static_vector<T, N>& items)
	{

		items.resize(0);
		while (true)
		{
			const auto name_ptr = ar.getNodeName();
			if (!name_ptr)
			{
				break;
			}
			T item;
			ar(item);
			//will throw on size mismatch
			items.push_back(item);
		}
	}
}

#endif