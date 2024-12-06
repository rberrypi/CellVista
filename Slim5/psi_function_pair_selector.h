#pragma once
#ifndef PSI_FUNCTION_PAIR_SELECTOR_H
#define PSI_FUNCTION_PAIR_SELECTOR_H
#include <QWidget>
#include "modulator_configuration.h"
namespace Ui
{
	class psi_function_pair_selector;
}
class psi_function_pair_selector final : public QWidget
{
	Q_OBJECT

	std::unique_ptr<Ui::psi_function_pair_selector> ui_;
	void psi_function_pair_update();
	void set_is_complete(bool is_complete);
public:
	explicit psi_function_pair_selector(QWidget *parent = Q_NULLPTR);
	[[nodiscard]] psi_function_pair get_psi_function_pair() const;
	virtual ~psi_function_pair_selector();
public slots:
	void set_psi_function_pair(const psi_function_pair& settings);
	void set_horizontal(bool horizontal);
	
signals:
	void psi_function_pair_changed(const psi_function_pair& settings);
};


#endif 
