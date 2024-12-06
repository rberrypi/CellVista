#include "stdafx.h"
#include "render_shifter_selector.h"

#include "qli_runtime_error.h"
#include "ui_render_shifter_selector.h"

void render_shifter_selector::update_render_shifter_selector()
{
	const auto values = get_render_shifter();
	emit ml_render_shifter(values);
}

render_shifter_selector::render_shifter_selector(QWidget* parent) : QWidget(parent)
{
	ui = std::make_unique< Ui::render_shifter_selector>();
	ui->setupUi(this);
	QObject::connect(ui->qsbTX, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &render_shifter_selector::update_render_shifter_selector);
	QObject::connect(ui->qsbTY, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &render_shifter_selector::update_render_shifter_selector);

}

render_shifter render_shifter_selector::get_render_shifter() const
{
	const auto tx = ui->qsbTX->value();
	const auto ty = ui->qsbTY->value();
	const render_shifter render_shifter(tx, ty);
	return render_shifter;
}

void render_shifter_selector::set_render_shifter(const render_shifter& remapper) const
{
	ui->qsbTX->setValue(remapper.tx);
	ui->qsbTY->setValue(remapper.ty);
#if _DEBUG
	{
		const auto what_we_got = get_render_shifter();
		if (!remapper.item_approx_equals(what_we_got))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

