#include "stdafx.h"
#include "per_modulator_saveable_settings_selector.h"
#include "device_factory.h"
#include "slm_pattern_model.h"
#include <QPainter>
#include "ui_per_modulator_saveable_settings_selector.h"
#include "qli_runtime_error.h"
#include <QCheckBox>
#include "device_factory.h"
per_modulator_saveable_settings_selector::per_modulator_saveable_settings_selector(QWidget* parent) : QGroupBox(parent), slm_id(-1), darkfield_mode_(darkfield_mode::dots), mode_(slm_mode::unset), block_update_hack(false)
{
	ui_ = std::make_unique<Ui::per_modulator_saveable_settings_selector>();
	ui_->setupUi(this);
	pattern_model = std::make_unique<per_pattern_modulator_settings_patterns_model>();
	ui_->pattern_table->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectRows);
	ui_->pattern_table->setModel(pattern_model.get());
	QObject::connect(ui_->pattern_table->selectionModel(), &QItemSelectionModel::currentRowChanged, [&](const QModelIndex& current, const QModelIndex& )
	{
		const auto pattern = current.row();
		emit clicked_pattern(pattern);
	});
	setFocusPolicy(Qt::FocusPolicy::ClickFocus);
	per_modulator_saveable_settings blank;
#if _DEBUG
	if (!blank.is_valid())
	{
		qli_runtime_error();
	}
#endif
	set_per_modulator_saveable_settings(blank);
	QObject::connect(ui_->four_frame_psi_settings, &four_frame_psi_settings_selector::four_frame_psi_settings_changed, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(pattern_model.get(), &per_pattern_modulator_settings_patterns_model::dataChanged, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(ui_->darkfield, &darkfield_pattern_settings_selector::darkfield_pattern_settings_changed, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(ui_->slim_beam_settings, &distorted_donut_selector::value_changed, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(ui_->illumination, &illumination_power_settings_selector::illumination_power_settings_changed, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(ui_->voltage_max, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	QObject::connect(ui_->chk_alignment, &QCheckBox::stateChanged, this, &per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings);
	this->installEventFilter(this);
	//
	
	{
		const auto settings = get_per_modulator_saveable_settings();
		set_valid_voltage(settings.valid_voltage());
	}
#if _DEBUG
	{
		if (!get_per_modulator_saveable_settings().is_valid())
		{
			qli_runtime_error();
		}
	}
#endif
}

void per_modulator_saveable_settings_selector::paintEvent(QPaintEvent* event)
{
	if (hasFocus())
	{
		QPainter painter(this);
		const QRectF rectangle(0, 0, width() - 1, height() - 1);
		painter.drawRect(rectangle);
	}
	return QGroupBox::paintEvent(event);
}

bool per_modulator_saveable_settings_selector::eventFilter(QObject* object, QEvent* event)
{
	if (event->type() == QEvent::KeyPress)
	{
		const auto* key_event = dynamic_cast<QKeyEvent*>(event);
		const auto key = key_event->key();
		switch (key)
		{
		case Qt::Key::Key_Down:
			ui_->slim_beam_settings->bump_donut(0, -1, 0, 0);
			return true;
		case Qt::Key::Key_Up:
			ui_->slim_beam_settings->bump_donut(0, 1, 0, 0);
			return true;
		case Qt::Key::Key_Left:
			ui_->slim_beam_settings->bump_donut((-1), 0, 0, 0);
			return true;
		case Qt::Key::Key_Right:
			ui_->slim_beam_settings->bump_donut(1, 0, 0, 0);
			return true;
		case Qt::Key::Key_W:
			ui_->slim_beam_settings->bump_donut(0, 0, 1, 0);
			return true;
		case Qt::Key::Key_S:
			ui_->slim_beam_settings->bump_donut(0, 0, -1, 0);
			return true;
		case Qt::Key::Key_R:
			ui_->slim_beam_settings->bump_donut(0, 0, 0, 1);
			return true;
		case Qt::Key::Key_F:
			ui_->slim_beam_settings->bump_donut(0, 0, 0, (-1));
			return true;
		default:;
		}
	}
	return QGroupBox::eventFilter(object, event);
}

void per_modulator_saveable_settings_selector::set_darkfield_mode(const darkfield_mode& mode)
{
	darkfield_mode_ = mode;
	/*
	const auto is_illumination = slm_id == per_modulator_saveable_settings::illumination_idx;
	const auto is_psi = darkfield_mode_settings::settings.at(darkfield_mode_).is_four_frame_psi;
	ui_->four_frame_psi_settings->setHidden(is_psi);
	*/
}

void per_modulator_saveable_settings_selector::set_slm_mode(const slm_mode& mode)
{
	mode_ = mode;
	ui_->four_frame_psi_settings->set_slm_mode(mode);
	ui_->grb_beam->setHidden(!((mode_ == slm_mode::slim) || (mode_ == slm_mode::darkfield)));
	ui_->grb_darkfield->setHidden(mode_ != slm_mode::darkfield);
	reload_modulator();
	const auto is_color = D ? D->has_a_color_camera() : true;
	for (auto i = 0; i < pattern_model->columnCount(); ++i)
	{
		const auto visible = this->pattern_model->is_visible(i, mode_, is_color);
		if (visible)
		{
			ui_->pattern_table->showColumn(i);
		}
		else
		{
			ui_->pattern_table->hideColumn(i);
		}
	}

}

void per_modulator_saveable_settings_selector::set_slm_id(const int id)
{
	slm_id = id;
	const auto slm_size = D->get_slm_dimensions().at(id);
	const auto is_illumination = id == per_modulator_saveable_settings::illumination_idx;
	const auto postfix = is_illumination ? tr("Illuminator") : tr("Modulator");
	setTitle(QString("SLM #%1 [%2,%3] %4").arg(id).arg(slm_size.width).arg(slm_size.height).arg(postfix));
	const auto is_retarder = D->has_retarders().at(id);
	ui_->voltage_max->setHidden(!is_retarder);
	ui_->lblvoltage->setHidden(!is_retarder);
	ui_->illumination->setHidden(!is_illumination);
}

per_modulator_saveable_settings_selector::~per_modulator_saveable_settings_selector() = default;

void per_modulator_saveable_settings_selector::reload_modulator()
{
	const auto selected_pattern = ui_->pattern_table->selectionModel()->currentIndex().row();
	if (selected_pattern >= 0)
	{
		const auto frame_data = D->get_slm_frames(selected_pattern).at(slm_id);
		if (!frame_data.frame_pointer)
		{
			qli_runtime_error("Should always have some SLM frame");
		}
		const auto pix_map = QPixmap::fromImage(QImage(frame_data.frame_pointer, frame_data.width, frame_data.height, QImage::Format_Grayscale8));
		ui_->lblSLMImage->setPixmap(pix_map);
	}
}

void per_modulator_saveable_settings_selector::set_pattern(const int pattern)
{
	ui_->pattern_table->selectRow(pattern);
	reload_modulator();
}

void per_modulator_saveable_settings_selector::set_per_modulator_saveable_settings(const per_modulator_saveable_settings & settings)
{
#if _DEBUG
	if (!settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	QSignalBlocker blk(*this);
	block_update_hack = true;
	secret_path = settings.file_path_basedir;
	set_modulator_configuration(settings);
	pattern_model->set_per_pattern_modulator_settings_patterns(settings.patterns);
	ui_->chk_alignment->setChecked(settings.is_alignment);
	block_update_hack = false;
#if _DEBUG
	{
		const auto what_we_got = this->get_per_modulator_saveable_settings();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void per_modulator_saveable_settings_selector::set_modulator_configuration_silent(const modulator_configuration & configuration)
{
	QSignalBlocker sb(*this);
	set_modulator_configuration(configuration);
}

void per_modulator_saveable_settings_selector::set_modulator_configuration(const modulator_configuration & configuration)
{
#if _DEBUG
	if (!configuration.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	ui_->four_frame_psi_settings->set_four_frame_psi_settings_silent(configuration.four_frame_psi);
	ui_->voltage_max->silentSetValue(configuration.voltage_max);
	set_valid_voltage(configuration.valid_voltage());
	ui_->slim_beam_settings->set_distorted_donut_silent(configuration);
	ui_->darkfield->set_darkfield_pattern_settings_silent(configuration);
	ui_->illumination->set_illumination_power_settings(configuration);
#if _DEBUG
	{
		const auto what_we_set = get_modulator_configuration();
		if (!what_we_set.item_approx_equals(configuration))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

modulator_configuration per_modulator_saveable_settings_selector::get_modulator_configuration() const
{
	const auto voltage_max = ui_->voltage_max->value();
	const auto beam_settings = ui_->slim_beam_settings->get_distorted_donut();
	const auto darkfield_pattern_settings = ui_->darkfield->get_darkfield_pattern_settings();
	const auto four_frame_psi_setting_holder = ui_->four_frame_psi_settings->get_four_frame_psi_settings();
	const auto illumination = ui_->illumination->get_illumination_power_settings();
	auto config = modulator_configuration(beam_settings, darkfield_pattern_settings, four_frame_psi_setting_holder, illumination, voltage_max);
#if _DEBUG
	{
		if (!config.is_valid())
		{
			qli_runtime_error();
		}
	}
#endif
	return config;
}

per_modulator_saveable_settings per_modulator_saveable_settings_selector::get_per_modulator_saveable_settings() const
{
	const auto patterns = pattern_model->get_pattern_modulator_settings_patterns();
	const auto modulator_configuration = get_modulator_configuration();
	const auto alignment = ui_->chk_alignment->isChecked();
	per_modulator_saveable_settings settings(modulator_configuration, patterns, alignment, secret_path);
#if _DEBUG
	{
		if (!settings.is_valid())
		{
			qli_runtime_error();
		}
	}
#endif
	return settings;
}

void per_modulator_saveable_settings_selector::set_valid_voltage(const bool is_valid)
{
	const auto* color_label = is_valid ? "" : "color: red;";
	ui_->voltage_max->setStyleSheet(color_label);
}

void per_modulator_saveable_settings_selector::update_per_modulator_saveable_settings()
{
	if (block_update_hack)
	{
		return;
	}
	const auto values = get_per_modulator_saveable_settings();
	set_valid_voltage(values.valid_voltage());
	emit per_modulator_saveable_settings_changed(values);
}

