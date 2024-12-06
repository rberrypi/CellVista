#include "stdafx.h"
#include "folder_line_edit.h"
#include <QMouseEvent>
#include <QFileDialog> 
#include "directory_validator.h"
#include <QStandardPaths>

folder_line_edit::folder_line_edit(QWidget* parent) : QLineEdit(parent)
{
	this->setValidator(new directory_validator);//pointer owned by the class?
}

void folder_line_edit::mouseDoubleClickEvent(QMouseEvent* e)
{
	if (e->button() == Qt::LeftButton)
	{
		if constexpr (!(IS_NIKON_APARTMENT_BUG))
		{
			const auto old = this->text();
			const auto old_directory = QDir(old).exists() ? old : QStandardPaths::standardLocations(QStandardPaths::DesktopLocation)[0];
			const auto filename = QFileDialog::getExistingDirectory(this, QString(), old_directory);
			if (!filename.isEmpty())
			{
				this->setText(filename);
			}
		}
	}
	QLineEdit::mouseDoubleClickEvent(e);
}