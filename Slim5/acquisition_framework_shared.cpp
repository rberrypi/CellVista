#include "stdafx.h"
#include "acquisition_framework.h"
#include <fstream>
#include "device_factory.h"
#include "save_device_state.h"
#include <iostream>
#include "compute_engine.h"
#include "render_widget.h"
#include <QDir>
#include "capture_modes.h"

void raw_io_work_meta_data::write_file_convention(const std::string& dir)
{
	const auto label = QString::fromStdString(filename_convention_header('_'));
	const auto filepath = QDir(QString::fromStdString(dir)).filePath(label);
	std::fstream in;
	in.open(filepath.toStdString(), std::fstream::out);
	in.close();
}

struct save_functors final : boost::noncopyable
{

	std::shared_ptr<compute_engine> compute;
	background_update_functors functor;
	explicit save_functors(const std::shared_ptr<compute_engine>& engine) : compute(engine)
	{
		functor = engine->phase_update_functors();
	}
	~save_functors()
	{
		compute->set_background_update_functors(functor);
	}
};

acquisition_meta_data acquisition_framework::capture_wrapper( capture_mode capture_mode, render_engine* engine)
{
	D->route.assert_valid();
	const save_device_state old(D->cameras, D.get(), D->scope.get());
	save_functors save(compute);
	// for the acquisition mode these functors will set
	compute->set_background_update_functors(D->get_background_update_functors());
	//
	const auto& output_directory = D->route.output_dir;
	raw_io_work_meta_data::write_file_convention(output_directory);
	const auto outfile_name = output_directory + "\\Capture_Log.csv";
	std::fstream log_output(outfile_name, std::fstream::out);
	write_capture_log_line_header(log_output) << std::endl;
	const auto log_to_console = D->io_show_cmd_progress;
	if (log_to_console)
	{
		write_capture_log_line_header(std::cout) << std::endl;
	}
	for (size_t i = 0; i < D->route.ch.size(); ++i)
	{
		auto channel = D->route.ch.at(i);
		const auto channel_path = output_directory + "\\channel_" + std::to_string(i) + ".json";
		channel.write(channel_path);
	}
	abort_capture = false;
	acquisition_meta_data meta_data;
	compute->flush_and_reset();
	compute->assert_that_outputs_have_been_serviced();
	const auto total_patterns = D->route.total_patterns(true);
	set_io_progress_total(total_patterns);
	set_capture_total(total_patterns);
	if (capture_mode == capture_mode::burst_capture_async_io)
	{
		meta_data = capture_burst_mode(old.pos_scope.scope_channel, log_output);
	}
	else
	{
		const auto& settings = capture_mode_settings::info.at(capture_mode);
		meta_data = capture(settings.async_capture, settings.async_io, log_to_console, old.pos_scope.scope_channel, log_output, engine);
	}
	//
	D->route.clear();
	compute->assert_that_outputs_have_been_serviced();
	set_io_progress_total(0);
	set_capture_total(0);
	//Check that progress has been correctly set
	return meta_data;
}
