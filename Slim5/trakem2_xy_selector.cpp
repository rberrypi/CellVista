#include "stdafx.h"
#include "trakem2_xy_selector.h"

#include "qli_runtime_error.h"
#include "ui_trakem2_xy_selector.h"

void trakem2_xy_selector::update_trakem2_xy()
{
	const auto trakem2_xy = get_trakem2_xy();
	emit trakem2_xy_changed(trakem2_xy);
}

trakem2_xy_selector::trakem2_xy_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::trakem2_xy_selector>();
	ui_->setupUi(this);
	for (auto* qsb : { ui_->qsbx,ui_->qsby })
	{
		connect(qsb, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &trakem2_xy_selector::update_trakem2_xy);
	}
}

trakem2_xy trakem2_xy_selector::get_trakem2_xy() const
{
	const auto x = ui_->qsbx->value();
	const auto y = ui_->qsby->value();
	return trakem2_xy(x, y);
}

void trakem2_xy_selector::set_trakem2_xy(const trakem2_xy& trakem2_xy)
{
	ui_->qsbx->setValue(trakem2_xy.x);
	ui_->qsby->setValue(trakem2_xy.y);
#if _DEBUG
	{
		const auto what_we_got = this->get_trakem2_xy();
		if (!what_we_got.approx_equal(trakem2_xy))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
