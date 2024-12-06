#pragma once
#ifndef DIR_VALIDATOR_H
#define DIR_VALIDATOR_H
#include <QValidator>
class directory_validator final : public QValidator
{
	Q_OBJECT
public:
	explicit directory_validator(QObject* parent = nullptr);
	State validate(QString& input, int& pos) const override;
};
#endif