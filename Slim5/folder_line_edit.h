#pragma once
#ifndef FOLDER_LINE_EDIT_H
#define FOLDER_LINE_EDIT_H
#include <QLineEdit> 
class folder_line_edit final : public QLineEdit
{
public:
	explicit folder_line_edit(QWidget* parent = Q_NULLPTR);
	void mouseDoubleClickEvent(QMouseEvent* e) override;
};
#endif