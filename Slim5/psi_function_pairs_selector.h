#pragma once
#ifndef PSI_FUNCTION_PAIRS_SELECTOR_H
#define PSI_FUNCTION_PAIRS_SELECTOR_H
#include <QWidget>
#include "modulator_configuration.h"
namespace Ui
{
	class psi_function_pairs_selector;
}

class psi_function_pair_selector;
class psi_function_pairs_selector final : public QWidget
{
	Q_OBJECT

	std::vector<psi_function_pair_selector*> psi_function_widgets_;
	void update_psi_function_pairs();
	std::unique_ptr<Ui::psi_function_pairs_selector> ui_;
	
public:
	explicit psi_function_pairs_selector(QWidget *parent = Q_NULLPTR);
	[[nodiscard]] psi_function_pairs get_psi_function_pairs() const;
	
public slots:
	void set_psi_function_pairs(const psi_function_pairs& settings);
	void set_horizontal(bool horizontal);
	
signals:
	void psi_function_pairs_changed(const psi_function_pairs& settings);
};


#endif 
