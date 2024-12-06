#include "stdafx.h"
#include "live_capture_engine.h"
#include <QSound>

void live_capture_engine::play_done_sound()
{
	const auto random_number = timestamp().count() % 75;
	const auto* const file = random_number == 0 ? ":/audio/done.wav" : ":/audio/En-ca-done.wav";
	QSound::play(file);
}
