#pragma once
#ifndef PATH_LOAD_SAVE_SELECTOR_H
#define PATH_LOAD_SAVE_SELECTOR_H
#include <QWidget>
namespace Ui {
	class path_load_save_selector;
}

class path_load_save_selector final : public QWidget
{
	Q_OBJECT

		std::unique_ptr<Ui::path_load_save_selector> ui_;
public:
	explicit path_load_save_selector(QWidget* parent = nullptr);
	[[nodiscard]] QString get_path();

public slots:
	void set_default_text(const QString& default_text);
	void set_text(const QString& text);
	void hide_save(bool save);

signals:
	void save_button_clicked();
	void load_button_clicked();
	void text_changed(const QString& text);
};

#endif