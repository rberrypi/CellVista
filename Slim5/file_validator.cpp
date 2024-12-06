#include "stdafx.h"
#include "file_validator.h"
//#include <QFileInfo>
#include <QDir>

file_validator::file_validator(QObject* parent) : QValidator(parent), file_prefix("file:///")
{
}

QValidator::State file_validator::validate(QString& input, int&) const
{
	input = input.trimmed();
	const auto has_it = input.contains(file_prefix);
	if (has_it)
	{
		return Intermediate;
	}
	input = QDir::toNativeSeparators(input);
	const QFileInfo f(input);
	const auto directory_exists = f.absoluteDir().exists();
	const auto is_a_directory = f.isDir();
	if (!directory_exists || is_a_directory)
	{
		return Invalid;
	}
	return Acceptable;
}

void file_validator::fixup(QString& input) const
{
	input.remove(file_prefix);
	//check if folder is valid!
}
