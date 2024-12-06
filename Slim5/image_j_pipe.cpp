#include "stdafx.h"
#include "image_j_pipe.h"
#include <QProcess>
#include <QFile>
#include <iostream>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
#include <Ice/Ice.h>
#include <ImageJPipeSlice.h>

template <class Archive>
void serialize(Archive & archive, image_j_pipe & cc)
{
	archive(
		cereal::make_nvp("imagej_exe_path", cc.imagej_exe_path)
	);
}

Ice::CommunicatorPtr ice_communicator;
QLI::ImageJInteropPrx ice_communicator_function_pointers;

void image_j_pipe::ice_safe_call(const std::function<void()>& function)
{
	for (auto i = 0; i < max_attempts_to_reconnect; ++i)
	{
		try
		{
			function();
			break;
		}
		catch (const Ice::Exception& exception)
		{
			Q_UNUSED(exception);
#if _DEBUG
			std::cout << exception << std::endl;
#endif
			setup_connection();
		}
	}
}

void image_j_pipe::send_segmentation(const unsigned char* map, const float* img_in, const frame_size& frame, const frame_meta_data& meta_data, const render_settings& render_settings, const std::string& name, bool live)
{
	const auto safe_call = [&]
	{
		//full_path
		std::pair<const Ice::Byte*, const Ice::Byte*> seg_data = { map, map + frame.n() };
		std::pair<const Ice::Float*, const Ice::Float*> map_data = { img_in, img_in + frame.n() };
		QLI::ImageJImageMetaData meta;
		meta.lutName = render_settings::luts[render_settings.display_lut].name;
		meta.name = name;
		//tdo fix display for RGB
		meta.displaymin = render_settings.ranges.front().min;
		meta.displaymax = render_settings.ranges.front().max;
		meta.pixelratio = meta_data.pixel_ratio;
		//
		QLI::segmentation_settings segmentation;
		segmentation.circmin = render_settings.segmentation_circ_min;
		segmentation.circmax = render_settings.segmentation_circ_max;
		segmentation.areamin = render_settings.segmentation_area_min;
		segmentation.areamax = render_settings.segmentation_area_max;
		segmentation.islive = live;
		ice_communicator_function_pointers->sendSegmentation(frame.width, frame.height, map_data, seg_data, meta, segmentation);
	};
	ice_safe_call(safe_call);
}

void image_j_pipe::send_image(const float* img, const frame_size& img_size, const frame_meta_data& meta_data, const render_settings& render_settings, const std::string& filename)
{
	const auto safe_call = [&]
	{
		auto start_ptr = img;
		auto end_ptr = img + img_size.n();
		std::pair<const Ice::Float*, const Ice::Float*> data = { start_ptr,end_ptr };
		QLI::ImageJImageMetaData meta;
		meta.lutName = render_settings::luts[render_settings.display_lut].name;
		meta.name = filename;
		meta.displaymin = render_settings.ranges.front().min;
		meta.displaymax = render_settings.ranges.front().max;
		meta.pixelratio = meta_data.pixel_ratio;
		ice_communicator_function_pointers->sendImage(img_size.width, img_size.height, data, meta);
	};
	ice_safe_call(safe_call);
}

void image_j_pipe::setup_connection()
{
	disconnect();
	const auto try_connect = [&] {
		try {
			ice_communicator = Ice::CommunicatorPtr();
			auto props = Ice::createProperties();
			props->setProperty("Ice.MessageSizeMax", "51200");
			Ice::InitializationData id;
			id.properties = props;
			ice_communicator = initialize(id);
			ice_communicator_function_pointers = QLI::ImageJInteropPrx::checkedCast(ice_communicator->stringToProxy("PhiOptics:default -h localhost -p 11230"));
			return true;
		}
		catch (...)
		{
			return false;
		}
	};
	const auto is_an_imagej_already_running = try_connect();
	if (!is_an_imagej_already_running)//For example a previous imageJ is open, so why the heck and not shuffle data into it?
	{
		const auto path = QString::fromStdString(imagej_exe_path);
		QFile imagej_file(path);
		if (!imagej_file.exists())
		{
			std::cout << "Can't find ImageJ executable: " << imagej_exe_path << std::endl;
			return;
		}
		process_handle_ = new  QProcess;
		process_handle_->start(path, QIODevice::Unbuffered);
		const auto wait_for_launch = (45000);
		process_handle_->waitForStarted(wait_for_launch);
		windows_sleep(ms_to_chrono(1000 * 10));
		//
		const auto status = try_connect();//For example a previous imageJ is open, so why the heck and not shuffle data into it?
		Q_UNUSED(status);
	}
}

void image_j_pipe::disconnect()
{
	if (process_handle_)
	{
		process_handle_->kill();
		process_handle_->deleteLater();//?
		process_handle_ = nullptr;
	}
	if (ice_communicator)
	{
		ice_communicator->destroy(); // throwing destructor?
	}
}

void image_j_pipe::set_image_j_exe_path(const QString& exe_path)
{
	imagej_exe_path = exe_path.toStdString();
	//disconnect();
	//setup_connection();
}

auto image_j_pipe_setting_path = "ImageJPipe.json";

image_j_pipe::image_j_pipe() : process_handle_(nullptr)
{
	std::ifstream configfile(image_j_pipe_setting_path);
	if (configfile.is_open())
	{
		cereal::JSONInputArchive archive(configfile);
		archive(*this);
	}
	else
	{
		std::cout << "Warning can't find device configuration file:" << image_j_pipe_setting_path << std::endl;
	}
	//Nuke an invalid entry
	if (!QFile(QString::fromStdString(imagej_exe_path)).exists())
	{
		imagej_exe_path = std::string();
	}
}

void image_j_pipe::save_pipe_path()
{
	std::ofstream os(image_j_pipe_setting_path);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
		std::cout << "Writing settings file to:" << image_j_pipe_setting_path << std::endl;
	}
	else
	{
		std::cout << "Warning can't write settings file to: " << image_j_pipe_setting_path << std::endl;
	}
}

image_j_pipe::~image_j_pipe()
{
	disconnect();
	save_pipe_path();
}