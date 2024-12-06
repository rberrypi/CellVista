#include "stdafx.h"
#include "slim_four.h"
#include "render_widget.h"
//#include "device_factory.h"
#include "render_container.h"
#include <QTabBar>
#include "write_debug_gpu.h"
#include <QDir>
#include "ui_slim_four.h"
#include "live_capture_engine.h"
#include "qt_layout_disable.h"

Q_DECLARE_METATYPE(display_settings::display_ranges);

slim_four::~slim_four()
{
	capture_engine->terminate_live_capture();
}

QTabBar* slim_four::get_bg_tab_bar() const
{
	const auto* central_layout = qobject_cast<QGridLayout*>(ui_->centralwidget->layout());
	const auto* inner_layout = central_layout->itemAt(0)->layout();
	auto* item = qobject_cast<QTabBar*>(inner_layout->itemAt(0)->widget());
	return item;
}

compute_and_scope_settings::background_frame_ptr slim_four::get_background_frame() const
{
	//maybe wrong because the actual background is stored in the live capture widget?
	const auto current_idx = get_contrast_idx();
	auto phase_bg = current_contrast_settings_.at(current_idx).background_;
	return phase_bg;
}

void slim_four::enable_bg_tabs(const bool enable) const
{
	//Note this doesn't clear nothing
	ui_->btn_clear_bg->setEnabled(enable);
	auto* tab_bar = get_bg_tab_bar();
	tab_bar->setTabEnabled(1, enable);
	tab_bar->setTabEnabled(2, enable);
}

void slim_four::setup_take_phase_background()
{
	QObject::connect(ui_->btn_set_bg,&QPushButton::clicked,capture_engine,&live_capture_engine::take_background);
	QObject::connect(ui_->btn_clear_bg,&QPushButton::clicked,capture_engine,&live_capture_engine::clear_background);
	//Tab Bar Changes
	const auto background_enabled = [&](const bool enable)
	{
		enable_bg_tabs(enable);
		auto* tab_bar = get_bg_tab_bar();
		const auto is_live = tab_bar->currentIndex() == 0;
		if (enable && is_live)
		{
			tab_bar->setCurrentIndex(2);
		}
		else
		{
			tab_bar->setCurrentIndex(0);
		}
	};
	QObject::connect(capture_engine,&live_capture_engine::background_enabled,background_enabled);
	//Note we ensure the gui changes at every live function call?
	QObject::connect(get_bg_tab_bar(), &QTabBar::currentChanged , this, &slim_four::live_compute_options_update);
}

live_compute_options slim_four::get_live_compute_options() const
{
	const auto compute_mode = [&]
	{
		const auto* tab_bar = get_bg_tab_bar();
		const auto current_index=tab_bar->currentIndex();
		const auto current_item = static_cast<live_compute_options::background_show_mode>(current_index);
		return current_item;
	}();
	const live_compute_options set(true, compute_mode);
	return set;
}

void slim_four::setup_render_widget_and_capture_engine()
{
	//Render Surface
	{
		{
			render_surface_ = new render_widget(compute);
			auto* tab_bar = new QTabBar();
			tab_bar->addTab("Live");
			tab_bar->addTab("Background");
			tab_bar->addTab("Background Subtracted");
			tab_bar->setCurrentIndex(0);
			tab_bar->setExpanding(false);
			const render_widget::ml_filter_function filter = [&](const float* input_array, const frame_size& frame)
			{
				const auto write_something = ui_->btnSnapML->isChecked();
				if (write_something)
				{
					static auto ml_snap = 0;
					const auto filename = QString("ML_Snapshot_%1.tif").arg(++ml_snap);
					//needs to be modified when syncing with Taha
					const auto output_directory = ui_->txtOutputDir->text();
					const auto path = QDir(output_directory).filePath(filename);
					const auto str_path = path.toStdString();
					write_debug_gpu(input_array, frame.width, frame.height, 1, str_path.c_str(), true);
					ui_->btnSnapML->setChecked(false);
				}
			};
			render_surface_->ml_filter = filter;
			auto* scroll_area = new render_container(render_surface_, this);
			auto* temp_layout = new QVBoxLayout;
			temp_layout->addWidget(tab_bar);
			temp_layout->addWidget(scroll_area);
			temp_layout->setAlignment(scroll_area, Qt::AlignHCenter);
			auto* layout = qobject_cast<QGridLayout*>(ui_->centralwidget->layout());
			layout->addLayout(temp_layout, 0, 0);
			enable_bg_tabs(false);
		}
		//probably should actually maximize?
		connect(this, &slim_four::repaint_render, [&] { render_surface_->render_later(); });
		connect(render_surface_, &render_widget::load_histogram, this, &slim_four::load_histogram);
		qRegisterMetaType<display_settings::display_ranges>();
		connect(render_surface_, &render_widget::load_auto_contrast_settings, this, &slim_four::load_auto_contrast_settings, Qt::QueuedConnection);
	}
	//Compute Engine
	{
		capture_engine = new live_capture_engine(render_surface_, compute, this);
		QObject::connect(ui_->btnFixSkip, &QPushButton::clicked, capture_engine, &live_capture_engine::fix_capture);
		QObject::connect(capture_engine, &live_capture_engine::set_capture_progress, this, &slim_four::set_capture_progress, Qt::ConnectionType::QueuedConnection);
		QObject::connect(capture_engine, &live_capture_engine::set_capture_total, this, &slim_four::set_capture_total, Qt::ConnectionType::QueuedConnection);
		QObject::connect(capture_engine, &live_capture_engine::set_io_progress, this, &slim_four::set_io_progress, Qt::ConnectionType::QueuedConnection);
		QObject::connect(capture_engine, &live_capture_engine::set_io_progress_total, this, &slim_four::set_io_progress_total, Qt::ConnectionType::QueuedConnection);
		QObject::connect(capture_engine, &live_capture_engine::set_io_buffer_progress, this, &slim_four::set_io_buffer_progress, Qt::ConnectionType::QueuedConnection);

		QObject::connect(this, &slim_four::stop_acquisition, capture_engine, &live_capture_engine::stop_acquisition);
		QObject::connect(ui_->btn_stop_acquisition, &QPushButton::clicked, this, &slim_four::stop_acquisition);
		QObject::connect(capture_engine, &live_capture_engine::gui_enable,this,&slim_four::set_gui_for_acquisition, Qt::ConnectionType::QueuedConnection);
	}
}

void slim_four::set_gui_for_acquisition(const bool enable_gui)
{
	ui_->btnFullInterface->setEnabled(enable_gui);
	ui_->slm_settings_file->setEnabled(enable_gui);
	ui_->txtOutputDir->setReadOnly(!enable_gui);
	enable_layout(ui_->bandpass_layout,enable_gui);
	enable_layout(ui_->grdContrast,enable_gui);
	ui_->cmb_camera_config->setEnabled(enable_gui);
	ui_->processing_quad->setEnabled(enable_gui);
	ui_->wdg_display_settings->setEnabled(enable_gui);
	ui_->wdg_phase_shift_exposures_and_delays->setEnabled(enable_gui);
	enable_layout(ui_->frame_microscope_new_xyz,enable_gui);
	ui_->wdg_light_path->setEnabled(enable_gui);
	ui_->frame_ml->setEnabled(enable_gui);
	ui_->frame_snapshot->setEnabled(enable_gui);
	ui_->btn_stop_acquisition->setEnabled(!enable_gui);
	//
	ui_->slm_settings_file->set_gui_for_acquisition(enable_gui);
	
}