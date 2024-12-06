#include "stdafx.h"
#include "ml_structs.h"

#include "qli_runtime_error.h"

ml_remapper_qli_v2_network::ml_remapper_qli_v2_network() : auxiliary_x1(std::numeric_limits<float>::quiet_NaN()), auxiliary_x2(std::numeric_limits<float>::quiet_NaN())
{

}

std::unordered_map<ml_remapper_file::ml_remapper_types, ml_remapper_file> ml_remapper_file::ml_remappers = {
	{ml_remapper_file::ml_remapper_types::off,ml_remapper_file()},

	{ml_remapper_file::ml_remapper_types::pass_through_test_engine,ml_remapper_file("Test Mapper","",frame_size(896,896),0,65536,0,1,-3,3,3.14,true,ml_remapper_qli_v2_network())},

	{ml_remapper_file::ml_remapper_types::glim_dapi_20x,ml_remapper_file("DAPI (Hilbert 20/0.8)","transposed_weight.h5",frame_size(),qQNaN(),qQNaN(),0,1,0,65535,3.14,false,ml_remapper_qli_v2_network(-3.1446402,2.2227864))},
	
	{ml_remapper_file::ml_remapper_types::glim_dapi_20x_480,ml_remapper_file("DAPI (Hilbert 20/0.8) SW480","dapi_pure_transposed_weight.h5",frame_size(),qQNaN(),qQNaN(),0,1,0,65535,3.14,false,ml_remapper_qli_v2_network(-0.87952256,1.501151))},

	{ml_remapper_file::ml_remapper_types::glim_dil_20x,ml_remapper_file("DIL (Hilbert 20/0.8)","DIL_psnr_27_transposed_weight.h5",frame_size(),qQNaN(),qQNaN(),0,1,0,65535,3.14,false,ml_remapper_qli_v2_network(-3.1446402,1.7271369))},

	{ml_remapper_file::ml_remapper_types::slim_dapi_10x,ml_remapper_file("DAPI (SLIM 10/0.3)","slim_dapi_32_psnr_transposed_weight.h5",frame_size(),qQNaN(),qQNaN(),0,1,0,65535,1.57,false,ml_remapper_qli_v2_network(-3.141475, 3.141475))},

	{ml_remapper_file::ml_remapper_types::dpm_slim,ml_remapper_file("DPM to SLIM","202000915_DPM_SLIM_weight_transposed.h5",frame_size(1776, 1760),-3.14,3.14,0,1,-3.14, 3.14,8.21,false,ml_remapper_qli_v2_network(-3.141475, 3.141475))},

	{ml_remapper_file::ml_remapper_types::viability, ml_remapper_file("Viability", "Viability_model_nchw.onnx", frame_size(832, 832), -3.14, 3.14, 0, 255, 0, 255, 3.20, true, ml_remapper_qli_v2_network(-3.141475, 3.141475), 4, true, true, true)}

//
//░░░░█─────────────█──▀──
//░░░░▓█───────▄▄▀▀█──────
//░░░░▒░█────▄█▒░░▄░█─────
//░░░░░░░▀▄─▄▀▒▀▀▀▄▄▀─────
//░░░░░░░░░█▒░░░░▄▀────────
//▒▒▒░░░░▄▀▒░░░░▄▀────────
//▓▓▓▓▒░█▒░░░░░█▄─────────
//█████▀▒░░░░░█░▀▄────────
//█████▒▒░░░▒█░░░▀▄───────
//███▓▓▒▒▒▀▀▀█▄░░░░█──────
//▓██▓▒▒▒▒▒▒▒▒▒█░░░░█─────
//▓▓█▓▒▒▒▒▒▒▓▒▒█░░░░░█────
//░▒▒▀▀▄▄▄▄█▄▄▀░░░░░░░█─
// about the right size to shove up
// popescu's ass

	//{ml_remapper_file::ml_remapper_types::hrslim_dapi_10x,ml_remapper_file("DAPI (HR SLIM 10/0.3)","hrslim_newest_ff_dapi_10x_32_psnr_transposed_weight.h5",frame_size(),qQNaN(),qQNaN(),0,1,196.0, 25732.0,1.54,false,ml_remapper_qli_v2_network(-3.141475, 3.141475))}
};

const  std::set<ml_remapper_file::ml_remapper_types> ml_remapper_file::mappers_to_prebake = { ml_remapper_file::ml_remapper_types::viability };



[[nodiscard]] frame_size ml_remapper_file::get_network_size() const
{
#if _DEBUG
	if (!network_size.is_valid())
	{
		qli_runtime_error();
	}
#endif
	return network_size;
}