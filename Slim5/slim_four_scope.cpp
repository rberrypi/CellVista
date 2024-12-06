#include "stdafx.h"


#include "slim_four.h"
#include "device_factory.h"
#include "scope.h"
#include "ui_slim_four.h"

void slim_four::setup_location_update() 
{
	scope_timer = new QTimer(this);
	scope_timer->start(500);
	const auto scope_update_functor = [&]()
	{
		const auto location = D->scope->get_state(true);
		emit microscope_state_changed(location);
	};
	connect(scope_timer, &QTimer::timeout, scope_update_functor);
	const auto update_position_functor = [&](const microscope_state& state)
	{
		ui_->qsbmicroscopeX->setValue(state.x);
		ui_->qsbmicroscopeY->setValue(state.y);
		ui_->qsbmicroscopeZ->setValue(state.z);
	};
	connect(this,&slim_four::microscope_state_changed,update_position_functor);	
}

void slim_four::setup_microscope_move()
{
	const auto move_functor = [&]
	{
		//todo there is a special reserved value in qt
		constexpr auto magic = static_cast<double>(-1.0);
		auto here = D->scope->get_state();
		const auto x = ui_->qsbmicroscopeSetX->value();
		here.x = x == ui_->qsbmicroscopeSetX->minimum() || x == magic ? here.x : x;
		const auto y = ui_->qsbmicroscopeSetY->value();
		here.y = y == ui_->qsbmicroscopeSetY->minimum() || y == magic ? here.y : y;
		const auto z = ui_->qsbmicroscopeSetZ->value();
		here.z = z == ui_->qsbmicroscopeSetZ->minimum() || z == magic ? here.z : z;
		D->scope->move_to(here, true);
	};
	{
		const auto modify_spin_box = [](QDoubleSpinBox* box, const double min, const double max)
		{
			box->setMinimum(min);
			box->setMaximum(max);
			const auto str = QString("[%1,%2]").arg(min).arg(max);
			box->setToolTip(str);
		};
		{
			const auto xy_lim_settings = D->scope->xy_drive->get_stage_limits();
			if (xy_lim_settings.valid)
			{
				const auto xy_limits = xy_lim_settings.xy;
				modify_spin_box(ui_->qsbmicroscopeSetX, xy_limits.left(), xy_limits.right());
				modify_spin_box(ui_->qsbmicroscopeSetY, xy_limits.top(), xy_limits.bottom());
			}
		}
		{
			const auto z_limit = D->scope->z_drive->get_z_drive_limits_internal();
			if (z_limit.valid)
			{
				modify_spin_box(ui_->qsbmicroscopeSetZ, z_limit.zee_min, z_limit.zee_max);//Radiant Zee Max
			}
		}
	}
	connect(ui_->btn_microscope_move, &QPushButton::clicked, move_functor);
}
