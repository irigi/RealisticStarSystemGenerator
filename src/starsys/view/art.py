"""Procedural artwork of stars, planets and moons (numpy only; Qt-independent).

:func:`render` draws one body as an RGBA sphere image. Everything is a deterministic function
of the body's physical description (the JSON dict of :mod:`starsys.export`) and an integer seed,
so the same body always looks the same and two bodies of the same class still differ (palette
shifts, band counts, continents, ice caps, storms, craters...).

The look follows physics where there is a basis for it and is decorative otherwise:

* stars: blackbody colour of Teff (CIE 1931 colour-matching fit of Wyman, Sloan & Shirley
  2013 integrated over the Planck function), limb darkening (linear, u = 0.6), granulation;
* giant planets: palette by the Sudarsky et al. (2000) cloud classes of the equilibrium
  temperature (I ammonia < 150 K, II water < 350 K, III cloudless < 800 K, IV alkali
  < 1400 K, V silicate clouds), banding (zonal jets; more bands for fast rotators);
* ice giants and sub-Neptunes: methane-blue, hazy when warm;
* rocky bodies by climate (temperate: oceans/continents/clouds/ice caps; frozen: ice;
  airless: cratered regolith; runaway: Venus-like cloud deck; hot: lava);
* moons: icy or rocky; strong tidal heating gives Io-like sulphur-volcanic surfaces.

Rings are not drawn: the generator has no ring model, and the viewer shows only what is generated.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------------- noise


class Noise3:
    """Seeded 3D value noise with quintic smoothing, and fractal sums of it."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.perm = np.concatenate([rng.permutation(256)] * 2).astype(np.int64)
        self.vals = rng.random(256) * 2.0 - 1.0

    def _h(self, i, j, k):
        p = self.perm
        return self.vals[p[p[p[i & 255] + (j & 255)] + (k & 255)]]

    def __call__(self, x, y, z):
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        fx, fy, fz = x - xi, y - yi, z - zi
        u, v, w = (f * f * f * (f * (f * 6 - 15) + 10) for f in (fx, fy, fz))
        h = self._h
        x00 = h(xi, yi, zi) * (1 - u) + h(xi + 1, yi, zi) * u
        x10 = h(xi, yi + 1, zi) * (1 - u) + h(xi + 1, yi + 1, zi) * u
        x01 = h(xi, yi, zi + 1) * (1 - u) + h(xi + 1, yi, zi + 1) * u
        x11 = h(xi, yi + 1, zi + 1) * (1 - u) + h(xi + 1, yi + 1, zi + 1) * u
        return (x00 * (1 - v) + x10 * v) * (1 - w) + (x01 * (1 - v) + x11 * v) * w

    def fbm(self, p, octaves=5, freq=1.0, gain=0.5, lac=2.03):
        x, y, z = p
        total, amp, norm = 0.0, 1.0, 0.0
        for o in range(octaves):
            total = total + amp * self(x * freq + 17.1 * o, y * freq - 5.3 * o, z * freq + 9.7 * o)
            norm += amp
            amp *= gain
            freq *= lac
        return total / norm

    def ridged(self, p, octaves=5, freq=1.0):
        x, y, z = p
        total, amp, norm = 0.0, 1.0, 0.0
        for o in range(octaves):
            total = total + amp * (1.0 - np.abs(self(x * freq + 3.1 * o, y * freq + 7.7 * o, z * freq - 1.3 * o)))
            norm += amp
            amp *= 0.5
            freq *= 2.1
        return total / norm


# ---------------------------------------------------------------------------------- colour


def _planck_xyz(teff: float) -> np.ndarray:
    lam = np.linspace(380e-9, 780e-9, 81)
    nm = lam * 1e9

    def g(x, mu, s1, s2):
        return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)

    xb = 1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2)
    yb = 0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1)
    zb = 1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8)
    b = 1.0 / (lam**5 * (np.exp(np.minimum(1.4388e-2 / (lam * max(teff, 500.0)), 700)) - 1.0))
    return np.array([np.sum(xb * b), np.sum(yb * b), np.sum(zb * b)])


def star_rgb(teff: float) -> np.ndarray:
    """sRGB (0-1) colour of a blackbody, brightest channel normalised to 1."""
    x, y, z = _planck_xyz(teff)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(m @ np.array([x, y, z]), 0, None)
    rgb = rgb / rgb.max()
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055).clip(0, 1)


def _hex(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _ramp(t, stops):
    """Piecewise-linear colour ramp; ``stops`` = [(t, '#rrggbb'), ...] with increasing t."""
    t = np.clip(t, stops[0][0], stops[-1][0])
    xs = np.array([s[0] for s in stops])
    cs = np.array([_hex(s[1]) if isinstance(s[1], str) else s[1] for s in stops])
    return np.stack([np.interp(t, xs, cs[:, c]) for c in range(3)], axis=-1)


def _shift(c, rng, hue=0.04, val=0.12):
    """Small random hue/brightness shift of a colour (keeps palettes recognisable)."""
    c = np.asarray(c, float)
    mean = c.mean()
    rot = rng.normal(0, hue)
    k = np.array([math.cos(rot * 6), math.cos(rot * 6 + 2.1), math.cos(rot * 6 - 2.1)])
    out = mean + (c - mean) * (1 + 0.5 * rot) + 0.1 * rot * (k - k.mean())
    return np.clip(out * (1 + rng.normal(0, val)), 0, 1)


# ---------------------------------------------------------------------------------- geometry


def _sphere(px: int, tilt: float, spin: float):
    """Pixel grid of a disc of diameter ``px``: mask, view normal z, and body-frame unit vectors."""
    c = (np.arange(px) + 0.5) / px * 2 - 1
    x, y = np.meshgrid(c, -c)
    r2 = x * x + y * y
    mask = r2 <= 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    # rotate view -> body: tilt about the view x axis... (pole tipped toward viewer by tilt)
    ct, st = math.cos(tilt), math.sin(tilt)
    yb, zb = y * ct + z * st, -y * st + z * ct
    cs, ss = math.cos(spin), math.sin(spin)
    xb, zb2 = x * cs + zb * ss, -x * ss + zb * cs
    return mask, x, y, z, (xb, yb, zb2)


def _light(x, y, z, direction=(-0.55, 0.35, 0.76), ambient=0.05):
    d = np.array(direction) / np.linalg.norm(direction)
    lam = np.clip(x * d[0] + y * d[1] + z * d[2], 0, 1)
    return ambient + (1 - ambient) * lam, lam


def _rgba(rgb, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    return (np.concatenate([np.clip(rgb, 0, 1), a], axis=-1) * 255).astype(np.uint8)


def _atmosphere(rgb, x, y, z, mask, colour, strength, lit):
    """Rim brightening of an atmosphere (stronger toward the limb, on the lit side)."""
    rim = (1 - z) ** 3 * strength * (0.25 + 0.75 * lit)
    return rgb * (1 - rim[..., None]) + np.asarray(colour) * rim[..., None]


def _halo(px_full, inner_frac, colour, strength, falloff=3.0):
    """Glow around a disc that occupies ``inner_frac`` of the image width."""
    c = (np.arange(px_full) + 0.5) / px_full * 2 - 1
    x, y = np.meshgrid(c, -c)
    r = np.sqrt(x * x + y * y) / inner_frac
    glow = strength * np.exp(-falloff * np.maximum(r - 1, 0)) * np.clip((1 / inner_frac - r) * 4, 0, 1)
    return np.broadcast_to(np.asarray(colour), x.shape + (3,)), glow


# ---------------------------------------------------------------------------------- bodies


def _star(px, teff, seed, kind):
    rng = np.random.default_rng(seed)
    n = Noise3(seed)
    if kind == "black hole":
        return _black_hole(px)
    inner = 0.72 if kind in ("star", "white dwarf") else 0.86
    d = max(int(px * inner), 4)
    mask, x, y, z, p = _sphere(d, 0.2, rng.random() * 6.28)
    col = star_rgb(teff)
    if kind == "brown dwarf":
        # Cool substellar atmospheres: banded, deep red-magenta (L) to dark violet (T/Y).
        lat = p[1] + 0.06 * n.fbm(tuple(3 * q for q in p), 4)
        bands = 0.5 + 0.5 * np.sin(lat * (10 + 6 * rng.random()) + 2 * n.fbm(tuple(2 * q for q in p), 3))
        warm = np.clip((teff - 500) / 1800, 0, 1)
        base = _hex("#2a1030") * (1 - warm) + _hex("#a8323a") * warm
        rgb = base * (0.7 + 0.5 * bands[..., None])
        lit, _ = _light(x, y, z, ambient=0.35)
        rgb = rgb * lit[..., None] + _hex("#ff5020") * 0.25 * warm * (z ** 2)[..., None]
        glow_c, glow_s = _hex("#ff3010") * warm, 0.35 * warm
    else:
        gran = n.fbm(tuple(14 * q for q in p), 4, gain=0.55)
        spots = np.clip(-n.fbm(tuple(2.5 * q for q in p), 3) - 0.3, 0, 1) * (teff < 6500)
        limb = 1 - 0.6 * (1 - z)  # linear limb darkening, u = 0.6
        col = np.clip(col.mean() + 2.2 * (col - col.mean()), 0, 1)  # display saturation (as in games)
        e = limb * (1 + 0.08 * gran) * (1 - 0.5 * spots)
        rgb = col * (0.45 + 0.75 * e)[..., None]
        rgb = rgb + (1 - rgb) * (0.45 * z ** 6)[..., None]  # bright core
        glow_c, glow_s = col, 0.9 if kind != "neutron star" else 1.2
        if kind == "neutron star":
            rgb = np.ones_like(rgb) * _hex("#cfe2ff")
    img = np.zeros((px, px, 4), np.uint8)
    hc, hs = _halo(px, d / px, glow_c, glow_s, falloff=2.2 if kind == "star" else 3.5)
    img[...] = _rgba(hc, hs)
    o = (px - d) // 2
    c = (np.arange(d) + 0.5) / d * 2 - 1
    xx, yy = np.meshgrid(c, -c)
    disc = _rgba(rgb, np.clip((1 - np.sqrt(xx * xx + yy * yy)) * d / 2, 0, 1))
    img[o:o + d, o:o + d] = over(disc, img[o:o + d, o:o + d])
    return img


def over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Alpha compositing of two straight-alpha RGBA uint8 images."""
    ta, ba = top[..., 3:4] / 255.0, bottom[..., 3:4] / 255.0
    oa = ta + ba * (1 - ta)
    rgb = (top[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) / np.maximum(oa, 1e-6)
    return np.concatenate([rgb, oa * 255], axis=-1).round().astype(np.uint8)


def _black_hole(px):
    c = (np.arange(px) + 0.5) / px * 2 - 1
    x, y = np.meshgrid(c, -c)
    r = np.sqrt(x * x + (y / 0.35) ** 2)
    disk = np.exp(-((r - 0.75) / 0.18) ** 2) * (np.abs(y) < 0.35)
    ring = np.exp(-((np.sqrt(x * x + y * y) - 0.42) / 0.035) ** 2)
    rgb = np.stack([np.ones_like(x), 0.6 + 0.0 * x, 0.25 + 0.0 * x], -1)
    alpha = np.clip(disk * 0.9 + ring, 0, 1)
    shadow = np.sqrt(x * x + y * y) < 0.38
    rgb[shadow] = 0
    alpha = np.where(shadow, 1.0, alpha)
    return _rgba(rgb * (0.5 + 0.5 * alpha[..., None]), alpha)


def _giant(n, rng, p, t_eq, rot_h, mass):
    """Banded giant: palette by Sudarsky class, band count by rotation."""
    if t_eq < 150:      # I: ammonia clouds (Jupiter, Saturn)
        pal = rng.choice(3)
        stops = [[(0, "#7a4a2a"), (0.35, "#c08a5a"), (0.6, "#e8d2b0"), (1, "#f6eedc")],
                 [(0, "#8a6a3a"), (0.4, "#c8a86a"), (0.7, "#e4d0a0"), (1, "#f2e6c8")],
                 [(0, "#6a3a22"), (0.3, "#b0603a"), (0.65, "#e0b890"), (1, "#f4e4d0")]][pal]
    elif t_eq < 350:    # II: water clouds, bright white
        stops = [(0, "#9aa4b0"), (0.5, "#dfe4ea"), (1, "#ffffff")]
    elif t_eq < 800:    # III: cloudless, Rayleigh-scattering blue
        stops = [(0, "#1d3a7a"), (0.5, "#3a6ab8"), (1, "#7ea8e0")]
    elif t_eq < 1400:   # IV: alkali absorption, dark
        stops = [(0, "#1a1a2a"), (0.5, "#3a3550"), (1, "#5e5a78")]
    else:               # V: silicate clouds, hot and glowing
        stops = [(0, "#3a1208"), (0.5, "#8a3a1a"), (1, "#d88a4a")]
    stops = [(t, _shift(_hex(c), rng, 0.03, 0.05)) for t, c in stops]
    nb = 6 + 10 * float(np.clip(30.0 / max(rot_h or 10.0, 3.0), 0.3, 1.5)) * (0.7 + 0.6 * rng.random())
    warp = n.fbm(tuple(2.0 * q for q in p), 5) * (0.12 + 0.12 * rng.random())
    turb = n.fbm((p[0] * 5, p[1] * 30, p[2] * 5), 4) * 0.05
    lat = p[1] + warp + turb
    phase = rng.random(8) * 6.28
    amps = rng.random(8)
    v = sum(amps[k] * np.sin(lat * nb * (0.5 + 0.25 * k) + phase[k]) for k in range(4)) / amps[:4].sum()
    v = 0.5 + 0.5 * v * (0.6 + 0.6 * rng.random())
    if t_eq >= 350:
        v = 0.5 + (v - 0.5) * 0.4  # weaker banding without condensate clouds
    rgb = _ramp(v, stops)
    # storms: a few ovals
    for _ in range(rng.integers(0, 4)):
        lat0, lon0 = rng.uniform(-0.6, 0.6), rng.uniform(0, 6.28)
        cx, cz = math.cos(lon0) * math.sqrt(1 - lat0**2), math.sin(lon0) * math.sqrt(1 - lat0**2)
        d2 = ((p[0] - cx) ** 2 + (p[2] - cz) ** 2) / (0.03 + 0.05 * rng.random()) + ((p[1] - lat0) / 0.05) ** 2
        s = np.exp(-d2 * 6)
        storm_c = _shift(_hex("#c0502a") if t_eq < 150 else _hex("#f0f0ff"), rng, 0.05, 0.1)
        rgb = rgb * (1 - s[..., None]) + storm_c * s[..., None]
    return rgb


def _ice_giant(n, rng, p, t_eq):
    if t_eq > 500:  # warm sub-Neptunes: hazy, photochemical
        base = _shift(_hex("#b8a888"), rng, 0.06, 0.08)
        hi = _shift(_hex("#e0d8c0"), rng)
    else:
        base = _shift(_hex(rng.choice(["#3a78c8", "#5ab0d0", "#4a90c0", "#6ac0c8"])), rng, 0.04, 0.08)
        hi = np.clip(base * 1.25 + 0.1, 0, 1)
    lat = p[1] + 0.05 * n.fbm(tuple(3 * q for q in p), 4)
    v = 0.5 + 0.5 * np.sin(lat * (5 + 4 * rng.random()) + rng.random() * 6) * (0.25 + 0.2 * rng.random())
    rgb = base * (1 - v[..., None]) + hi * v[..., None]
    if rng.random() < 0.4:  # a dark spot (Neptune)
        lat0, lon0 = rng.uniform(-0.5, 0.3), rng.uniform(0, 6.28)
        cx, cz = math.cos(lon0), math.sin(lon0)
        s = np.exp(-(((p[0] - cx * 0.9) ** 2 + (p[2] - cz * 0.9) ** 2) / 0.02 + ((p[1] - lat0) / 0.06) ** 2) * 4)
        rgb = rgb * (1 - 0.5 * s[..., None])
    return rgb


def _rocky(n, rng, p, climate, t_surf, t_eq, mass, water_lost):
    h = n.fbm(tuple(1.6 * q for q in p), 6, gain=0.52)
    detail = n.fbm(tuple(9 * q for q in p), 4)
    lat = np.abs(p[1])
    clouds = None
    if climate in ("temperate", "marginal"):
        ocean_frac = float(np.clip(rng.normal(0.6 if climate == "temperate" else 0.3, 0.2), 0.05, 0.95))
        if water_lost:
            ocean_frac *= 0.3
        sea = np.quantile(h, ocean_frac)
        land = h > sea
        elev = np.clip((h - sea) / (h.max() - sea + 1e-9), 0, 1)
        warm = np.clip(((t_surf or 288) - 260) / 60, 0, 1)
        veg = _shift(_hex(rng.choice(["#3a6a2a", "#4a6a30", "#2e5a3a", "#6a5a2a", "#5a3a5a"])), rng, 0.05, 0.1)
        desert = _shift(_hex("#b89a6a"), rng, 0.04, 0.08)
        dry = np.clip(0.5 + 0.8 * detail + (0.5 - lat) * 0.4 * (climate == "marginal"), 0, 1)
        land_c = veg * (1 - dry[..., None]) + desert * dry[..., None]
        land_c = land_c * (1 - 0.35 * elev[..., None]) + _hex("#8a8070") * (0.35 * elev[..., None])
        deep, shallow = _shift(_hex("#0e2a5a"), rng, 0.03, 0.08), _shift(_hex("#2a6a9a"), rng, 0.03, 0.08)
        depth = np.clip((sea - h) * 6, 0, 1)
        sea_c = shallow * (1 - depth[..., None]) + deep * depth[..., None]
        rgb = np.where(land[..., None], land_c, sea_c)
        cap = 0.95 - 0.35 * (1 - warm) - 0.1 * rng.random() + 0.06 * detail
        ice = lat > cap
        rgb = np.where(ice[..., None], _hex("#eef4ff"), rgb)
        clouds = 0.35 + 0.25 * rng.random()
    elif climate == "frozen":
        base = _shift(_hex(rng.choice(["#dfe8f0", "#d8dce0", "#e8e0d8", "#c8d8e8"])), rng, 0.03, 0.05)
        cracks = np.clip(1 - n.ridged(tuple(4 * q for q in p), 4) * 1.15, 0, 1) ** 3
        rgb = base * (0.85 + 0.15 * h[..., None]) * (1 - 0.35 * cracks[..., None])
        # darker exposed terrain / tholin-stained plains (Pluto, Triton)
        stain = _shift(_hex(rng.choice(["#8a6a5a", "#6a6a72", "#a08060", "#5a5048"])), rng, 0.05, 0.1)
        patch = np.clip((h - np.quantile(h, 0.4 + 0.5 * rng.random())) * 5, 0, 1)
        rgb = rgb * (1 - 0.8 * patch[..., None]) + stain * (0.8 * patch[..., None])
    elif climate == "runaway greenhouse":
        lat_w = p[1] + 0.15 * n.fbm(tuple(2.5 * q for q in p), 5)
        v = 0.5 + 0.3 * np.sin(lat_w * 8 + 3 * n.fbm(tuple(1.5 * q for q in p), 3))
        rgb = _ramp(v, [(0, _shift(_hex("#b08a40"), rng)), (1, _shift(_hex("#f0e0b0"), rng))])
    elif (t_eq or 0) > 900 or climate == "hot":
        cracks = n.ridged(tuple(3 * q for q in p), 5)
        hot = np.clip(((t_eq or 1000) - 700) / 1500, 0.2, 1)
        crust = _shift(_hex("#2a2420"), rng, 0.02, 0.1) * (0.8 + 0.4 * h[..., None])
        lava = np.clip((cracks - 0.72) * 6 * hot, 0, 1)
        rgb = crust * (1 - lava[..., None]) + _hex("#ff6a10") * lava[..., None]
    else:  # airless or thin atmosphere: cratered regolith
        base = _shift(_hex(rng.choice(["#8a8580", "#9a8a78", "#7a7470", "#a07858", "#8a7a6a"])), rng, 0.04, 0.1)
        rgb = base * (0.75 + 0.35 * h[..., None] + 0.1 * detail[..., None])
        rgb = _craters(rgb, rng, p, int(20 + 40 * rng.random()))
    return rgb, clouds


def _craters(rgb, rng, p, k):
    for _ in range(k):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = 0.04 + 0.18 * rng.random() ** 3
        d = np.sqrt((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 + (p[2] - v[2]) ** 2) / r
        rim = np.exp(-((d - 1) / 0.15) ** 2) * 0.25
        floor = (d < 1) * 0.12
        rgb = rgb * (1 + rim - floor)[..., None]
    return rgb


def _moon(n, rng, p, comp, flux):
    h = n.fbm(tuple(2 * q for q in p), 5)
    if flux is not None and flux > 1.0:  # Io-like (Io ~2 W/m^2)
        base = _shift(_hex("#d8c050"), rng, 0.04, 0.08)
        rgb = base * (0.85 + 0.25 * h[..., None])
        spots = np.clip(n.fbm(tuple(7 * q for q in p), 3) - 0.35, 0, 1) * 3
        rgb = rgb * (1 - spots[..., None]) + _hex("#7a2a10") * spots[..., None]
        if flux > 20:  # super-Io
            glow = np.clip(n.ridged(tuple(4 * q for q in p), 4) - 0.6, 0, 1) * 3
            rgb = rgb * (1 - glow[..., None]) + _hex("#ff7020") * glow[..., None]
        return rgb
    if comp == "ice":
        base = _shift(_hex(rng.choice(["#d8d4cc", "#c8c0b0", "#e0e4e8", "#b8a890"])), rng, 0.04, 0.08)
        lines = np.clip(1 - n.ridged(tuple(3 * q for q in p), 3) * 1.1, 0, 1) ** 4 * rng.random()
        rgb = base * (0.8 + 0.25 * h[..., None]) * (1 - 0.5 * lines[..., None])
        return _craters(rgb, rng, p, int(rng.integers(5, 30)))
    base = _shift(_hex(rng.choice(["#8a8580", "#7a7068", "#9a9088"])), rng, 0.04, 0.1)
    return _craters(base * (0.75 + 0.35 * h[..., None]), rng, p, int(rng.integers(15, 50)))


def render(body: dict, px: int, seed: int) -> np.ndarray:
    """RGBA uint8 image (px x px) of a body given as a :mod:`starsys.export` dict.

    The body's role is recognised from its keys: stars have ``teff``, planets ``type``, moons
    ``composition``.
    """
    px = max(int(px), 4)
    rng = np.random.default_rng(seed)
    n = Noise3(seed)
    if "teff" in body:
        return _star(px, float(body.get("teff") or 3000), seed, body.get("kind", "star"))
    tilt = math.radians(float(body.get("obliquity_deg") or 0) * 0.6) + rng.normal(0, 0.15)
    mask, x, y, z, p = _sphere(px, tilt, rng.random() * 6.28)
    lit, lam = _light(x, y, z)
    atm_c, atm_s, clouds, night = None, 0.0, None, None
    if "composition" in body:
        rgb = _moon(n, rng, p, body.get("composition"), body.get("heat_flux_w_m2"))
        flux = body.get("heat_flux_w_m2") or 0
        if flux > 20:
            night = 0.5
    else:
        kind, t_eq = body.get("type"), float(body.get("t_eq_k") or 300)
        if kind == "giant":
            rgb = _giant(n, rng, p, t_eq, body.get("rotation_h"), body.get("mass"))
            atm_c, atm_s = rgb.mean(axis=(0, 1)) * 0.6 + 0.4, 0.35
            if t_eq > 1400:
                night = min((t_eq - 1400) / 1500, 0.8)
        elif kind == "envelope":
            rgb = _ice_giant(n, rng, p, t_eq)
            atm_c, atm_s = np.clip(rgb.mean(axis=(0, 1)) + 0.25, 0, 1), 0.45
        else:
            climate = str(body.get("climate", "")).split(" (")[0]
            climate = {"thin or no atmosphere": "airless"}.get(climate, climate)
            climate = "hot" if t_eq > 900 and climate in ("airless", "unlit", "") else climate
            rgb, clouds = _rocky(n, rng, p, climate, body.get("t_surface_illustrative_k"), t_eq,
                                 body.get("mass"), (body.get("t_runaway_myr") or 0) > 100)
            if climate in ("temperate", "marginal"):
                atm_c, atm_s = _hex("#8ab8ff"), 0.6
            elif climate == "runaway greenhouse":
                atm_c, atm_s = _hex("#f0e0b0"), 0.5
            elif climate == "frozen" and (body.get("mass") or 0) > 0.3:
                atm_c, atm_s = _hex("#c8d8ff"), 0.3
            if climate == "hot" or t_eq > 900:
                night = 0.6
    if clouds:
        c = n.fbm((p[0] * 3 + 11, p[1] * 5, p[2] * 3), 5, gain=0.55)
        cov = np.clip((c - (0.5 - clouds)) * 3.5, 0, 0.9)
        rgb = rgb * (1 - cov[..., None]) + cov[..., None]
    shaded = rgb * lit[..., None]
    if night:  # self-luminous night side (lava, hot giants)
        emis = rgb * np.array([1.0, 0.45, 0.2]) * night
        shaded = np.maximum(shaded, emis * (1 - lam[..., None]) * 0.8)
    if atm_c is not None:
        shaded = _atmosphere(shaded, x, y, z, mask, atm_c, atm_s, lam)
    # anti-aliased edge
    c = (np.arange(px) + 0.5) / px * 2 - 1
    xx, yy = np.meshgrid(c, -c)
    edge = np.clip((1 - np.sqrt(xx * xx + yy * yy)) * px / 2, 0, 1)
    return _rgba(shaded, edge)


__all__ = ["Noise3", "render", "star_rgb"]
