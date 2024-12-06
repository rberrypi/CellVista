#pragma once
#ifndef FILE_LINE_EDIT_H
#define FILE_LINE_EDIT_H
#include <QLineEdit> 
class file_line_edit final : public QLineEdit
{
	Q_OBJECT

public:
	explicit file_line_edit(QWidget* parent = nullptr);
	void mouseDoubleClickEvent(QMouseEvent* e) override;
public slots:
	void double_click_prompt();
};
#endif