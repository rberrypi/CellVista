#include "stdafx.h"
#include "psi_function_pair_selector.h"
#include "ui_psi_function_pair_selector.h"
#include "qli_runtime_error.h"
psi_function_pair_selector::psi_function_pair_selector(QWidget *parent ) : QWidget(parent)
{
	ui_=std::make_unique<Ui::psi_function_pair_selector>();
	ui_->setupUi(this);
	for (auto* box : {ui_->bot,ui_->top,ui_->constant})
	{
		psi_function_pair::spin_box_settings.style_spin_box(box);
		QObject::connect(box,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&psi_function_pair_selector::psi_function_pair_update);
	}
	set_horizontal(false);
	set_is_complete(get_psi_function_pair().is_complete());
}

void psi_function_pair_selector::psi_function_pair_update()
{
	const auto pair =  get_psi_function_pair();
	set_is_complete(pair.is_complete());
	emit psi_function_pair_changed(pair);
}

void psi_function_pair_selector::set_is_complete(const bool is_complete)
{
	const auto* color_label = is_complete ? "" : "color: red;";
	for (auto item : {ui_->bot,ui_->top,ui_->constant})
	{
		item->setStyleSheet(color_label);		
	}
}

void psi_function_pair_selector::set_horizontal(const bool horizontal)
{
	const auto layout = horizontal ? QBoxLayout::QBoxLayout::LeftToRight : QBoxLayout::TopToBottom;
	ui_->verticalLayout->setDirection(layout);
}

void psi_function_pair_selector::set_psi_function_pair(const psi_function_pair& settings)
{
	ui_->bot->setValue(settings.bot);
	ui_->top->setValue(settings.top);
	ui_->constant->setValue(settings.constant);
#if _DEBUG
	{
		const auto what_we_got  = get_psi_function_pair();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

psi_function_pair_selector::~psi_function_pair_selector()= default;

psi_function_pair  psi_function_pair_selector::get_psi_function_pair() const
{
	const auto top = ui_->top->value();
	const auto bot = ui_->bot->value();
	const auto constant = ui_->constant->value();
	const psi_function_pair pair(top, bot, constant);
	return pair;
}
