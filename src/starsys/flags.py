"""Per-planet extrapolation flags (bit mask). Re-exported by :mod:`starsys.planets`."""

FLAG_COLOUR = 1  # host colour outside the calibrated range: bp-rp < 0.5 (~F3 and hotter),
                 # or > 1.7 when the M-dwarf extension is off
FLAG_YOUNG = 2  # system younger than 100 Myr: atmospheric loss still under way
FLAG_MULTIPLE = 4  # host has a companion inside 200 au (suppression and dynamical limits from
                   # MK21 and HW99 applied); without the binary model, any companion
FLAG_SUBSTELLAR = 8  # host below 0.1 Msun: occurrence from disk-mass scaling, no data
FLAG_GIANT_HOST = 16  # giant planet around a host outside the giant calibration's 0.2-3 Msun
FLAG_BRIDGE = 32  # small planet between HM-C's P_max and the snow line: interpolated, not measured
FLAG_COLD_EXTRAP = 64  # small planet beyond the snow line outside the microlensing-measured mass ratios
                       # (log q < -5.2), separations (> 4 snow-line radii) or host masses (0.1-1 Msun)
FLAG_CIRCUMBINARY = 128  # orbits the inner binary (P-type): single-star models outside the HW99 limit
FLAG_EVOLVED = 256  # host is a subgiant, giant or white dwarf: main-sequence planets, engulfed and expanded
FLAG_PULSAR = 512  # neutron-star planet: placeholder rate and properties
FLAG_POST_INTERACTION = 1024  # a component of the system overfilled its Roche lobe (no binary evolution modelled)
FLAG_SHORT_PERIOD = 2048  # planet inside HM-C's inner period limit (Hsu+2019 / Sanchis-Ojeda+2014 rates)
FLAG_HZ_EXTRAP = 4096  # habitable-zone limits extrapolated: >10% of the flux from stars outside 2600-7200 K
FLAG_KOZAI = 8192  # eccentricity and inclination from Kozai-Lidov cycles driven by a stellar companion
FLAG_MOON = 16384  # moons (starsys.moons): theory-driven, no exomoon detections to fit

FLAG_NAMES = {FLAG_COLOUR: "colour", FLAG_YOUNG: "young", FLAG_MULTIPLE: "multiple", FLAG_SUBSTELLAR: "substellar",
              FLAG_GIANT_HOST: "giant_host", FLAG_BRIDGE: "bridge", FLAG_COLD_EXTRAP: "cold_extrapolated",
              FLAG_CIRCUMBINARY: "circumbinary", FLAG_EVOLVED: "evolved", FLAG_PULSAR: "pulsar",
              FLAG_POST_INTERACTION: "post_interaction", FLAG_SHORT_PERIOD: "short_period",
              FLAG_HZ_EXTRAP: "hz_extrapolated", FLAG_KOZAI: "kozai", FLAG_MOON: "moon"}
