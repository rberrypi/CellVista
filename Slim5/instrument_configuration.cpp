#include "stdafx.h"
#include "instrument_configuration.h"
bool scope_location_xyz::is_valid() const
{
	const auto good = isfinite(x) && isfinite(y) && isfinite(z);
	return good;
}
