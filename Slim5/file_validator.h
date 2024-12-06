#pragma once
#ifndef FILE_VALIDATOR_H
#define FILE_VALIDATOR_H
#include <QValidator>
class file_validator final : public QValidator
{
	Q_OBJECT
public:
	explicit file_validator(QObject* parent = nullptr);
	State validate(QString& input, int& pos) const override;
	void fixup(QString& input) const  override;
	const char* file_prefix;
};
#endif