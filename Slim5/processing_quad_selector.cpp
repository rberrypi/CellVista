#include "stdafx.h"
#include "processing_quad_selector.h"
#include <QStandardItemModel>
#include "device_factory.h"
#include <QSignalBlocker> 
#include "ui_processing_quad_selector.h"
#include <QComboBox>
#include "qli_runtime_error.h"

processing_quad_selector::processing_quad_selector(QWidget* parent) : QWidget(parent)
{
	// Demosaic -> Retrieval -> Processing & Denoise
	ui = std::make_unique<Ui::processing_quad_selector>();
	ui->setupUi(this);
	switch_layout(false);
	//Step 1 Set to a reasonable default state
	const processing_quad default_settings;// what the heck is this?
	{
		ui->cmbDemosaic->addItem(QString::fromStdString(demosaic_setting::info.at(default_settings.demosaic).label), QVariant::fromValue(default_settings.demosaic));
		ui->cmbRetrieval->addItem(QString::fromStdString(phase_retrieval_setting::settings.at(default_settings.retrieval).label), QVariant::fromValue(default_settings.retrieval));
		ui->cmbProcessing->addItem(QString::fromStdString(phase_processing_setting::settings.at(default_settings.processing).label), QVariant::fromValue(default_settings.processing));
		ui->cmbDenoise->addItem(QString::fromStdString(denoise_setting::settings.at(default_settings.denoise).label), QVariant::fromValue(default_settings.denoise));
	}
	{
		auto max_content=0;
		ui->cmbDemosaic->clear();
		for (auto const& demosaic_mode : D->system_demosaic_modes())
		{
			const auto mode = demosaic_setting::info.at(demosaic_mode).label;
			const auto enum_value = QVariant::fromValue(demosaic_mode);
			const auto text = QString::fromStdString(mode);
			max_content=std::max(max_content,text.size());
			ui->cmbDemosaic->addItem(text, enum_value);
		}
		ui->cmbDemosaic->setMinimumContentsLength(max_content);
	}
	hidden_if_empty(ui->cmbDemosaic);
	//Step 2 Connect signals
	connect(ui->cmbDemosaic, qOverload<int>(&QComboBox::currentIndexChanged), this, &processing_quad_selector::update_retrieval);
	connect(ui->cmbRetrieval, qOverload<int>(&QComboBox::currentIndexChanged), this, &processing_quad_selector::update_processing_and_denoise);
	for (auto* cmb : { ui->cmbDemosaic,ui->cmbRetrieval,ui->cmbProcessing,ui->cmbDenoise })
	{
		QObject::connect(cmb, qOverload<int>(&QComboBox::currentIndexChanged), [&, cmb]
		{
			hidden_if_empty(cmb);
		});
		connect(cmb, qOverload<int>(&QComboBox::currentIndexChanged), this, &processing_quad_selector::update_processing_quad);
	}
	//Step 3 Flow data
	update_retrieval();
	update_processing_and_denoise();
}

void processing_quad_selector::update_processing_quad()
{
	const auto value = get_quad();
#if _DEBUG
	if (!value.is_supported_quad())
	{
		qli_runtime_error();
	}
#endif
	emit processing_quad_changed(value);
}

void processing_quad_selector::switch_layout(const bool to_grid)
{
	const auto direction = to_grid ? QBoxLayout::TopToBottom : QBoxLayout::LeftToRight;
	ui->main_layout->setDirection(direction);
}

void processing_quad_selector::update_retrieval()
{
	const auto demosaic = get_demosaic();
	const auto& retrieval_modes = demosaic_setting::info.at(demosaic).supported_retrieval_modes;
	const auto old_retrieval = ui->cmbRetrieval->currentData();
	{
		//this is a dirty hack
		
		const auto potential_color_camera = D->has_a_color_camera();
		QSignalBlocker lk(ui->cmbRetrieval);
		ui->cmbRetrieval->clear();
		for (auto const& retrieval : retrieval_modes)
		{
			if (retrieval==phase_retrieval::custom_patterns)
			{
				continue;
			}
			const auto& settings = phase_retrieval_setting::settings.at(retrieval);
			if (potential_color_camera && !settings.has_color_paths)
			{
				continue;
			}
			const auto enum_value = QVariant::fromValue(retrieval);
			const auto text = QString::fromStdString(settings.label);
			ui->cmbRetrieval->addItem(text, enum_value);
		}
	}
	set_if_found(ui->cmbRetrieval, old_retrieval);
	hidden_if_empty(ui->cmbRetrieval);
	assert_valid_indexes();
}

void processing_quad_selector::update_processing_and_denoise()
{
	const auto retrieval = get_retrieval();
	const auto& phase_retrieval_setting = phase_retrieval_setting::settings.at(retrieval);
	//
	{
		const auto processing_modes = phase_retrieval_setting.supported_processing_modes;
		const auto old_processing = ui->cmbProcessing->currentData();
		{
			QSignalBlocker lk(ui->cmbProcessing);
			ui->cmbProcessing->clear();
			for (auto const& processing : processing_modes)
			{
				const auto& settings = phase_processing_setting::settings.at(processing);
				const auto enum_value = QVariant::fromValue(processing);
				const auto text = QString::fromStdString(settings.label);
				ui->cmbProcessing->addItem(text, enum_value);
			}
		}
		set_if_found(ui->cmbProcessing, old_processing);
		hidden_if_empty(ui->cmbProcessing);
		assert_valid_indexes();
	}
	//
	{
		const auto denoise_modes = phase_retrieval_setting.supported_denoise_modes;
		const auto old_denoise = ui->cmbDenoise->currentData();
		QSignalBlocker lk(ui->cmbDenoise);
		{
			ui->cmbDenoise->clear();
			for (auto const& denoise : denoise_modes)
			{
				const auto& settings = denoise_setting::settings.at(denoise);
				const auto enum_value = QVariant::fromValue(denoise);
				const auto text = QString::fromStdString(settings.label);
				ui->cmbDenoise->addItem(text, enum_value);
			}
		}
		set_if_found(ui->cmbDenoise, old_denoise);
		hidden_if_empty(ui->cmbDenoise);
		assert_valid_indexes();
	}
}

void processing_quad_selector::assert_valid_indexes() const
{
#if _DEBUG
	{
		const auto retrieval_invalid = ui->cmbRetrieval->currentIndex();
		const auto processing_invalid = ui->cmbProcessing->currentIndex();
		const auto demosaic_invalid = ui->cmbDemosaic->currentIndex();
		if (retrieval_invalid < 0 || processing_invalid < 0 || demosaic_invalid < 0)
		{
			qli_invalid_arguments();
		}
	}
#endif
}

processing_quad processing_quad_selector::get_quad() const
{
	assert_valid_indexes();
	const auto retrieval_mode = get_retrieval();
	const auto processing_mode = get_processing();
	const auto demosaic = get_demosaic();
	const auto denoise = get_denoise();
	const auto quad = processing_quad(retrieval_mode, processing_mode, demosaic, denoise);
	if (!quad.is_supported_quad())
	{
		qli_gui_mismatch();
	}
	return quad;
}

void processing_quad_selector::set_processing(const processing_quad& quad) const
{
#if _DEBUG
	{
		if (!quad.is_supported_quad())
		{
			qli_runtime_error();
		}
	}
#endif
	{
		const auto idx = ui->cmbDemosaic->findData(QVariant::fromValue(quad.demosaic));
		if (idx >= 0)
		{
			ui->cmbDemosaic->setCurrentIndex(idx);
		}
		else
		{
			qli_runtime_error();
		}
	}
	{
		const auto idx = ui->cmbRetrieval->findData(QVariant::fromValue(quad.retrieval));
		if (idx >= 0)
		{
			ui->cmbRetrieval->setCurrentIndex(idx);
		}
		else
		{
			qli_runtime_error();
		}
	}
	//
	{
		const auto idx = ui->cmbProcessing->findData(QVariant::fromValue(quad.processing));
		if (idx >= 0)
		{
			ui->cmbProcessing->setCurrentIndex(idx);
		}
		else
		{
			qli_runtime_error();
		}
	}
	//
	{
		const auto idx = ui->cmbDenoise->findData(QVariant::fromValue(quad.denoise));
		if (idx >= 0)
		{
			ui->cmbDenoise->setCurrentIndex(idx);
		}
		else
		{
			qli_runtime_error();
		}
	}
#if _DEBUG
	{
		const auto what_we_got = get_quad();
		if (!(what_we_got == quad))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void processing_quad_selector::toggle_everything_but_camera(const bool enable) const
{
	//http://stackoverflow.com/a/21740341/314290
	const auto camera = phase_retrieval::camera;
	const auto* model = qobject_cast<const QStandardItemModel*>(ui->cmbRetrieval->model());
	for (auto i = 0; i < ui->cmbRetrieval->count(); ++i)
	{
		auto type_as_variant = ui->cmbRetrieval->itemData(i);
		const auto type = type_as_variant.value<phase_retrieval>();
		const auto item = model->item(i);
		if (type != camera)
		{
			item->setFlags(enable ? Qt::ItemIsSelectable | Qt::ItemIsEnabled : item->flags() & ~(Qt::ItemIsSelectable | Qt::ItemIsEnabled));
			// visually disable by graying out - works only if combobox has been painted already and palette returns the wanted color
			item->setData(enable ? QVariant(), Qt::ForegroundRole // clear item data in order to use default color (hark a wild comma operator!
				: ui->cmbRetrieval->palette().color(QPalette::Disabled, QPalette::Text));
		}
	}
}

void processing_quad_selector::set_if_found(QComboBox* cmb, const QVariant& data)
{
	const auto idx = cmb->findData(data);
	if (idx >= 0)
	{
		cmb->setCurrentIndex(idx);
	}
};

void processing_quad_selector::hidden_if_empty(QComboBox* cmb)
{
	cmb->setHidden(cmb->count() < 2);
}

phase_processing processing_quad_selector::get_processing() const
{
	return ui->cmbProcessing->currentData().value<phase_processing>();
}

phase_retrieval processing_quad_selector::get_retrieval() const
{
	return ui->cmbRetrieval->currentData().value<phase_retrieval>();
}

demosaic_mode processing_quad_selector::get_demosaic() const
{
	return ui->cmbDemosaic->currentData().value<demosaic_mode>();
}

denoise_mode processing_quad_selector::get_denoise() const
{
	return ui->cmbDenoise->currentData().value<denoise_mode>();
}