#include "stdafx.h"
#include "file_line_edit.h"
#include <QMouseEvent>
#include <QFileDialog> 
#include "file_validator.h"
#include <QStandardPaths>

file_line_edit::file_line_edit(QWidget* parent) : QLineEdit(parent)
{
	this->setValidator(new file_validator);//pointer owned by the class?
}
void file_line_edit::double_click_prompt()
{
	if constexpr (!(IS_NIKON_APARTMENT_BUG))
	{
		const auto opts = QFileDialog::DontConfirmOverwrite;
		const auto old = this->text();
		const QFileInfo f(old);
		const auto old_directory = f.exists() ? old : QStandardPaths::standardLocations(QStandardPaths::DesktopLocation)[0];
		const auto filename = QFileDialog::getSaveFileName(this, "Enter", old_directory, QString(), nullptr, opts);
		if (!filename.isEmpty())
		{
			setText(filename);
		}
	}
}
void file_line_edit::mouseDoubleClickEvent(QMouseEvent* e)
{
	if (e->button() == Qt::LeftButton)
	{
		double_click_prompt();
	}
	QLineEdit::mouseDoubleClickEvent(e);
}