#include "compute_engine.h"
#include "thrust_resize.h"
#include "ml_transformer.h"
compute_engine::compute_engine(const frame_size& output_size) : disconnect_queues_(false)
{
	//output_elements
	const auto output_elements = output_size.n();
	if (output_elements == 0)
	{
		qli_invalid_arguments();
	}
	//
	const auto max_retrieval_patterns =  processing_quad::max_retrieval_input_frames();
	input_frames.resize(max_retrieval_patterns);
	for (auto& pre_allocate : input_frames)
	{
		thrust_safe_resize(pre_allocate, output_elements);
	}
	//const auto muh_ram = 15;//wtf? needs to come from something more intellectual?
	//max_denoise_cycles = std::min(max_denoise_cycles, muh_ram);

	const auto max_denoise_cycles = denoise_setting::max_denoise_patterns();
	const auto memory_limit = 20;
	const auto max_denoise_patterns = std::max(max_denoise_cycles * max_retrieval_patterns, memory_limit);
	auto const max_output_frames = 3;//should be maximal of the cycle, also wtf
	output_frames.resize(max_output_frames + max_denoise_patterns);
	const auto max_n = static_cast<int>(std::max(output_size.width, output_size.height));//tood make sure they are the same
	filter_buffer_ptr = &filter_buffer;
	shifter_buffer_ptr = &shifter_buffer;
	decomplexify_buffer_ptr = &decomplexify_buffer;
	for (auto& ptr : { filter_buffer_ptr ,decomplexify_buffer_ptr,shifter_buffer_ptr })
	{
		thrust_safe_resize(*ptr, output_elements);
	}
	pre_allocate_fourier_filters(output_size);
	if (processing_quad::has_dpm())
	{
		pre_allocate_dpm_structs(output_size);
	}
	ml_transformer::pre_bake();
	const auto info = get_cuda_memory_info();
	//for example cufft or nvvp aren't prealocated?
	const auto pad = 30;
	const auto frames_possible = static_cast<int>((info.free_byte / ((output_elements) * sizeof(float)))) - pad;
	if (frames_possible < output_frames.size())
	{
		std::cout << "Warning only have enough memory for " << frames_possible << " frames. Certain denoise modes may not work." << std::endl;
		output_frames.resize(frames_possible);
#if _DEBUG
		qli_runtime_error("Don't be running tests with a computer that don't have enough RAM!");
#endif
	}
	for (auto&& pre_allocate : output_frames)
	{
		thrust_safe_resize(pre_allocate, output_elements);
		output_free_queue.push_back(&pre_allocate);
	}
	//
	cuda_memory_debug(__FILE__, __LINE__);
	//
	master_alias_check({});
}