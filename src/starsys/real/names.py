"""Star-name normalisation, so that "alpha Centauri A", "alf Cen A", "Alp1Cen" (Bright Star
Catalogue) and "Rigil Kentaurus" can be matched against the catalogue identifiers.

A name becomes a key: lower case, Greek letters and constellations spelled out in a fixed short
form, catalogue prefixes (HD, HIP, GJ, TIC, Gaia DR2/DR3/EDR3, 2MASS, ...) followed by their
number without spaces, punctuation dropped.
"""

from __future__ import annotations

import re

GREEK = {
    "alpha": "alf", "alp": "alf", "alf": "alf", "beta": "bet", "bet": "bet", "gamma": "gam", "gam": "gam",
    "delta": "del", "del": "del", "epsilon": "eps", "eps": "eps", "zeta": "zet", "zet": "zet", "eta": "eta",
    "theta": "tet", "the": "tet", "tet": "tet", "iota": "iot", "iot": "iot", "kappa": "kap", "kap": "kap",
    "lambda": "lam", "lam": "lam", "mu": "mu", "nu": "nu", "xi": "ksi", "ksi": "ksi", "omicron": "omi",
    "omi": "omi", "pi": "pi", "rho": "rho", "sigma": "sig", "sig": "sig", "tau": "tau", "upsilon": "ups",
    "ups": "ups", "phi": "phi", "chi": "chi", "psi": "psi", "omega": "ome", "ome": "ome",
}

# IAU constellations: abbreviation -> genitive.
CONSTELLATIONS = {
    "and": "andromedae", "ant": "antliae", "aps": "apodis", "aqr": "aquarii", "aql": "aquilae", "ara": "arae",
    "ari": "arietis", "aur": "aurigae", "boo": "bootis", "cae": "caeli", "cam": "camelopardalis",
    "cnc": "cancri", "cvn": "canum venaticorum", "cma": "canis majoris", "cmi": "canis minoris",
    "cap": "capricorni", "car": "carinae", "cas": "cassiopeiae", "cen": "centauri", "cep": "cephei",
    "cet": "ceti", "cha": "chamaeleontis", "cir": "circini", "col": "columbae", "com": "comae berenices",
    "cra": "coronae australis", "crb": "coronae borealis", "crv": "corvi", "crt": "crateris", "cru": "crucis",
    "cyg": "cygni", "del": "delphini", "dor": "doradus", "dra": "draconis", "equ": "equulei",
    "eri": "eridani", "for": "fornacis", "gem": "geminorum", "gru": "gruis", "her": "herculis",
    "hor": "horologii", "hya": "hydrae", "hyi": "hydri", "ind": "indi", "lac": "lacertae", "leo": "leonis",
    "lmi": "leonis minoris", "lep": "leporis", "lib": "librae", "lup": "lupi", "lyn": "lyncis", "lyr": "lyrae",
    "men": "mensae", "mic": "microscopii", "mon": "monocerotis", "mus": "muscae", "nor": "normae",
    "oct": "octantis", "oph": "ophiuchi", "ori": "orionis", "pav": "pavonis", "peg": "pegasi", "per": "persei",
    "phe": "phoenicis", "pic": "pictoris", "psc": "piscium", "psa": "piscis austrini", "pup": "puppis",
    "pyx": "pyxidis", "ret": "reticuli", "sge": "sagittae", "sgr": "sagittarii", "sco": "scorpii",
    "scl": "sculptoris", "sct": "scuti", "ser": "serpentis", "sex": "sextantis", "tau": "tauri",
    "tel": "telescopii", "tri": "trianguli", "tra": "trianguli australis", "tuc": "tucanae",
    "uma": "ursae majoris", "umi": "ursae minoris", "vel": "velorum", "vir": "virginis", "vol": "volantis",
    "vul": "vulpeculae",
}
_GENITIVE = {v.replace(" ", ""): k for k, v in CONSTELLATIONS.items()}

# Catalogue prefixes whose number follows: normalised to a fixed spelling.
_PREFIXES = {
    "hd": "hd", "hip": "hip", "hr": "hr", "gj": "gj", "gl": "gj", "gliese": "gj", "tic": "tic", "toi": "toi",
    "koi": "koi", "kic": "kic", "kepler": "kepler", "k2": "k2", "wasp": "wasp", "hat-p": "hatp", "hatp": "hatp",
    "lhs": "lhs", "lp": "lp", "ltt": "ltt", "nltt": "nltt", "wolf": "wolf", "ross": "ross", "bd": "bd",
    "cd": "cd", "cpd": "cpd", "lalande": "lalande", "wd": "wd", "2mass": "2mass", "lspm": "lspm", "g": "g",
}
_GAIA = re.compile(r"^gaia\s*(?:e?dr[23])?\s*(\d{6,})$")


def _tokens(name: str) -> list[str]:
    s = name.strip().lower()
    s = s.replace("’", "'").replace("'s", "s")
    s = re.sub(r"^(\*{1,2}|v\*|\*\*)\s+", "", s)  # SIMBAD object-type prefixes
    s = re.sub(r"^-+", "", s)  # hierarchy markers of the 20-pc census
    s = s.replace("_", " ")
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+\-]", s)


def key(name: str) -> str:
    """Normalised lookup key of a star name ("" when nothing is left)."""
    if not isinstance(name, str) or not name.strip():
        return ""
    raw = name.strip().lower()
    m = _GAIA.match(re.sub(r"\s+", " ", raw))
    if m:
        return "gaia" + m.group(1)
    toks = _tokens(name)
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        # Bright Star Catalogue style "Alp1Cen" / "58Alp Lyr": Greek letter glued to digits.
        if t in GREEK:
            out.append(GREEK[t])
            if nxt.isdigit() and len(nxt) <= 2:  # superscript index, e.g. alf1 Cen, omicron02 Eri
                out.append(str(int(nxt)))
                i += 1
            i += 1
            continue
        if t in _GENITIVE or (t + nxt) in _GENITIVE:
            if t + nxt in _GENITIVE and t not in _GENITIVE:
                out.append(_GENITIVE[t + nxt])
                i += 2
                continue
            out.append(_GENITIVE[t])
            i += 1
            continue
        if t in ("star", "the"):
            i += 1
            continue
        out.append(t)
        i += 1
    # Two-word genitives ("canis majoris") are handled above only when adjacent tokens join to a
    # key; join everything and map known prefixes.
    s = "".join(out)
    s = s.replace("hatp", "hatp")
    for p, n in _PREFIXES.items():
        if s.startswith(p.replace("-", "")) and s[len(p.replace("-", "")):][:1].isdigit():
            s = n + s[len(p.replace("-", "")):]
            break
    return s


def bsc_name_key(name: str) -> str:
    """Key of a Bright Star Catalogue name such as "58Alp Lyr", "Alp1Cen" or "33    Psc".

    The Flamsteed number is dropped when a Bayer letter is present (both names map to the
    same star; the Bayer one is the common one)."""
    if not isinstance(name, str):
        return ""
    m = re.match(r"^\s*(\d*)\s*([A-Za-z]{2,3})?(\d?)\s*([A-Za-z]{3})\s*$", name)
    if not m:
        return key(name)
    flam, greek, idx, con = m.groups()
    if greek and greek.lower() in GREEK:
        return GREEK[greek.lower()] + idx + con.lower()
    return (flam or "") + con.lower()


def bsc_flamsteed_key(name: str) -> str:
    """Flamsteed key ("61cyg") of a Bright Star Catalogue name, or "" without a number."""
    if not isinstance(name, str):
        return ""
    m = re.match(r"^\s*(\d+)\s*(?:[A-Za-z]{2,3}\d?)?\s*([A-Za-z]{3})\s*$", name)
    return m.group(1) + m.group(2).lower() if m else ""


def split_component(name_key: str) -> tuple[str, str]:
    """Split a trailing component letter: "alfcena" -> ("alfcen", "a"); "gj667c" -> ("gj667", "c").

    Only a single trailing a-d letter after a digit or a constellation is split; "" otherwise."""
    m = re.match(r"^(.*?(?:\d|" + "|".join(CONSTELLATIONS) + r"))([a-d]{1,2})$", name_key)
    if m and m.group(1):
        return m.group(1), m.group(2)
    return name_key, ""
