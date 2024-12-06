#include "stdafx.h"
#include "directory_validator.h"
#include <QDir>


directory_validator::directory_validator(QObject* parent) : QValidator(parent)
{
	//
}

QValidator::State directory_validator::validate(QString& input, int&) const
{
	input = input.trimmed();
	const auto exists = QDir(input).exists();
	input = QDir::toNativeSeparators(input);
	return exists ? Acceptable : Invalid;
}