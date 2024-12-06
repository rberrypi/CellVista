#include <npp.h>
#include "npp_error_check.h"
#include "npp_histogram.h"
#include <thrust/device_vector.h>
#include "thrust_resize.h"
#include <numeric>
#include "cuda_error_check.h"
#include "frame_size.h"
//NPP_SUCCESS
struct cuda_histogram_npp_impl
{
	//becuase msvc 2013 sucks with thread local storage?
	thrust::device_vector<Npp8u> pDeviceBuffer;
	std::array<thrust::device_vector<Npp32s>, 3> histDevice;
	unsigned long long n_old;
	int samples_per_pixel_old;
	cuda_histogram_npp_impl()
	{
		n_old = std::numeric_limits<size_t>::max();
		samples_per_pixel_old = -1;
	}
};

void cuda_npp::calc_histogram(histogram_info& info, const unsigned char* img_d, int n, int samples_per_pixel, const display_settings::display_ranges& range, bool is_auto_contrast) const
{
	const NppiSize oSizeROI = { n, 1 }; // full image
	if (samples_per_pixel == 1)
	{
		if (impl_->n_old != n)
		{
			CUDA_DEBUG_SYNC();
			int n_device_buffer_size;
			NPP_SAFE_CALL(nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, level_count, &n_device_buffer_size));//if size change_reallocate
			impl_->pDeviceBuffer.resize(n_device_buffer_size);

		}
		NPP_SAFE_CALL(nppiHistogramEven_8u_C1R(img_d, samples_per_pixel * n * sizeof(unsigned char), oSizeROI, thrust_safe_get_pointer(impl_->histDevice[0], 256), level_count, 0, bin_count, thrust::raw_pointer_cast(impl_->pDeviceBuffer.data())));
		//
		//write_debug_gpu(impl_->histDevice[0], impl_->histDevice[0].size(), 1, 1, "Test.tif", true);
	}
	else if (samples_per_pixel == 3)
	{
		int n_levels[3] = { level_count ,level_count , level_count };
		Npp32s n_lower_level[3] = { 0,0,0 };
		Npp32s n_upper_level[3] = { bin_count ,bin_count ,bin_count };
		if (impl_->n_old != n)
		{
			int n_device_buffer_size;
			NPP_SAFE_CALL(nppiHistogramEvenGetBufferSize_8u_C3R(oSizeROI, n_levels, &n_device_buffer_size));//if size change_reallocate
			impl_->pDeviceBuffer.resize(n_device_buffer_size);
		}
		Npp32s* p_hist[3] = { thrust_safe_get_pointer(impl_->histDevice[0],256) ,thrust_safe_get_pointer(impl_->histDevice[1],256) ,thrust_safe_get_pointer(impl_->histDevice[2],256) };
		NPP_SAFE_CALL(nppiHistogramEven_8u_C3R(img_d, samples_per_pixel * n * sizeof(unsigned char), oSizeROI, p_hist, n_levels, n_lower_level, n_upper_level, thrust::raw_pointer_cast(impl_->pDeviceBuffer.data())));
	}
	else
	{
		qli_not_implemented();
	}
	//Transfer data back
	info.samples_per_pixel = samples_per_pixel;
	for (auto i = 0; i < samples_per_pixel; ++i)
	{
		auto ptr = info.histogram_channels[i].data();
		thrust::copy(impl_->histDevice[i].begin(), impl_->histDevice[i].end(), ptr);
	}
	impl_->samples_per_pixel_old = samples_per_pixel;
	impl_->n_old = n;
	//Fill her up, maybe GPU this?
	//todo move to the GPU!!!
	for (auto frame_idx = 0; frame_idx < samples_per_pixel; ++frame_idx)
	{
		auto&& channel = info.histogram_channels[frame_idx];
		//Get std_dev
		const auto input_sum = std::accumulate(channel.begin(), channel.end(), 0);
		auto mean = 0.0f;
		for (auto i = 0; i < channel.size(); i++)
		{
			mean += static_cast<float>(channel[i]) * i;
		}
		mean = mean / input_sum;
		auto var = 0.0f;
		for (auto i = 0; i < channel.size(); i++)
		{
			const auto val = channel[i] * 1.0f * i * i;
			var += val;
		}
		var = var / input_sum;
		const auto stddev_raw = sqrt(var - mean * mean);
		//Get Thresholds
		/*
		auto interp = [](const int idx_a,const float v_a,const int idx_b,const float v_b, auto target_value)
		{
			const auto m = (v_b - v_a)/(idx_b-idx_a);
			const auto b = v_a + m*idx_a;
			//y=mx+b -> x=(y-b)/m
			return (target_value - b) / m;
		};
		*/
		//seriously, wtf is this shit
		static const auto auto_threshold = 5000;//vaguely from ImageJ, somehow this number works  well...
		const auto threshold = n / auto_threshold;
		float bot_idx = 0, top_idx = 0, mid_idx = 0;
		const auto discard_const = n / 10;
		for (auto i = 0; i < channel.size(); i++)
		{
			auto& val = channel[i];
			if ((val > threshold) && (val < discard_const))
			{
				bot_idx = i;
				break;
			}
		}
		for (int i = channel.size() - 1; i >= 0; i--)//unsugned arithemeti is dangeerous
		{
			auto& val = channel[i];
			if ((val > threshold) && (val < discard_const))
			{
				top_idx = i;
				break;
			}
		}
		const auto middle_threshold = n / 2;
		float cumsum = 0;//warning this can overflow, so we do float (2^20)*(2^8)(megapixels)
		for (auto i = 0; i < channel.size(); i++)
		{
			cumsum += channel[i];
			if (cumsum > middle_threshold)
			{
				mid_idx = i;
				break;
			}
		}
		//
		const auto max = range[frame_idx].max;
		const auto min = range[frame_idx].min;
		auto&& meta_data = info.info[frame_idx];
		meta_data.standard_deviation = (max - min) * (stddev_raw / (1.0 * channel.size() - 1));
		//when median is higher than top that means you might have saturation 
		meta_data.median = min + (max - min) * (mid_idx / (1.0 * channel.size() - 1));
		meta_data.bot_idx = bot_idx;
		meta_data.top_idx = top_idx;
		meta_data.bot = min + (max - min) * (bot_idx / (1.0 * channel.size() - 1));
		meta_data.top = min + (max - min) * (top_idx / (1.0 * channel.size() - 1));
	}
}

cuda_npp::cuda_npp()
{
	impl_ = new cuda_histogram_npp_impl;
	for (auto&& array : impl_->histDevice)
	{
		thrust_safe_resize(array, bin_count);
	}
}

cuda_npp::~cuda_npp()
{
	delete impl_;
}

cuda_npp::cuda_npp(cuda_npp&& a) noexcept
{
	impl_ = a.impl_;
}

cuda_npp::cuda_npp(const cuda_npp&)
{
	//or fuck it and just make some new buffers
	impl_ = new cuda_histogram_npp_impl;//this might be bad, did something leak?
}

cuda_npp& cuda_npp::operator = (cuda_npp a) //note: pass by value and let compiler do the magics
{
	impl_ = std::move(a.impl_);
	return *this;
}
