#include "stdafx.h"
#include "path_load_save_selector.h"
#include "ui_path_load_save_selector.h"

path_load_save_selector::path_load_save_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::path_load_save_selector>();
	ui_->setupUi(this);
	QObject::connect(ui_->btnSave, &QPushButton::clicked, this, &path_load_save_selector::save_button_clicked);
	QObject::connect(ui_->btnLoad, &QPushButton::clicked, [&]
	{
		if (ui_->txtLoad->text().isEmpty())
		{
			ui_->txtLoad->double_click_prompt();
		}
		else
		{
			emit load_button_clicked();
		}
	});
	QObject::connect(ui_->txtLoad, &QLineEdit::textChanged, [&](const QString& text)
	{
		ui_->btnSave->setEnabled(!text.isEmpty());
	});
	QObject::connect(ui_->txtLoad, &QLineEdit::textChanged, this, &path_load_save_selector::text_changed);
	ui_->btnSave->setEnabled(false);
}

QString path_load_save_selector::get_path()
{
	return ui_->txtLoad->text();
}

void path_load_save_selector::set_default_text(const QString& default_text)
{
	ui_->txtLoad->setPlaceholderText(default_text);
}

void path_load_save_selector::set_text(const QString& text)
{
	ui_->txtLoad->setText(text);
}

void path_load_save_selector::hide_save(const bool save)
{
	ui_->btnSave->setHidden(save);
}



