"""Physical constants and unit conversions.

SysSim catalogs store planet masses in solar masses and planet radii in solar
radii; we convert to Earth units using the same constants as ExoplanetsSysSim.jl
(``earth_mass``/``earth_radius`` in ``src/constants.jl``).
"""

from __future__ import annotations

# ExoplanetsSysSim.jl constants (in solar units), copied from its src/constants.jl.
EARTH_MASS_MSUN = 3.986004e14 / 1.3271244e20  # G*M_earth / G*M_sun, ~3.0035e-6
EARTH_RADIUS_RSUN = 6.3781e6 / 6.9566e8  # IAU 2015 equatorial R_earth / R_sun

G_SI = 6.67430e-11
MSUN_KG = 1.98840987e30
RSUN_M = 6.957e8
AU_M = 1.495978707e11
DAY_S = 86400.0
YEAR_DAYS = 365.25


def period_to_a_au(period_days, mstar_msun):
    """Semi-major axis (AU) from Kepler's third law, neglecting planet mass."""
    return (mstar_msun * (period_days / YEAR_DAYS) ** 2) ** (1.0 / 3.0)


def a_au_to_period(a_au, mstar_msun):
    """Orbital period (days) from semi-major axis, neglecting planet mass."""
    return YEAR_DAYS * (a_au**3 / mstar_msun) ** 0.5
