#include "stdafx.h"
#include "slim_four.h"
#include "material_picker.h"
#include "qli_runtime_error.h"
#include "ui_slim_four.h"

void slim_four::setup_material_settings()
{
	const auto open_material_settings = [&]
	{
		if (material_picker_dialog_ == nullptr)
		{
			qli_not_implemented();
			/*
			const auto processing = ui_->processing_quad->get_quad().processing;
			const auto current_material = get_settings_file();
			material_picker_dialog_ = new material_picker(processing, current_material, this);// no parents?
			connect(material_picker_dialog_, &QMainWindow::destroyed, [&]
			{
				material_picker_dialog_ = nullptr;
			});
			const auto lambda = [&](const processing_quad& pair)
			{
				if (material_picker_dialog_)
				{
					material_picker_dialog_->set_phase_processing(pair.processing);
				}
			};
			connect(ui_->processing_quad, &processing_quad_selector::value_changed, lambda);
			connect(material_picker_dialog_, &material_picker::material_info_updated, this,&slim_four::set_material_info);
			*/
		}
		material_picker_dialog_->showNormal();
		material_picker_dialog_->raise();
	};
	connect(ui_->btnMaterialSettings, &QPushButton::clicked, open_material_settings);
}

void slim_four::set_material_info(const material_info& info)
{
	hidden_properties.material_info = info;
	if (material_picker_dialog_)
	{
		qli_not_implemented();
	}
}

void slim_four::close_material_picker() const
{
	if (material_picker_dialog_)
	{
		material_picker_dialog_->close();
	}
}
