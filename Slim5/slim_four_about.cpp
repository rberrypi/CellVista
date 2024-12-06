#include "stdafx.h"
#include "slim_four.h"
#include <QMessageBox>

void slim_four::show_about()
{//might not update
	auto text = QString("Software: Mikhail E. Kandel \nHardware: Catalin Chiritescu\nDeveloped at the University of Illinois at Urbana-Champaign with financial support from Phi Optics Inc. \nkandel3@illinois.edu\\nBody: %1\nStage: %2\n").arg(BODY_TYPE).arg(STAGE_TYPE);
#if _DEBUG
	text.append("Brought to you by Stice Wine Co.\n \'Its drinkable!\' says Dr. Martha Gillette\n");
#endif
	text.append(QString("Build Date: %1 @ %2 ").arg(__DATE__).arg(__TIME__));
	QMessageBox::about(nullptr, "Info", text);
}
