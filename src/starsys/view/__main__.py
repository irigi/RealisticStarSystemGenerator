"""``python -m starsys.view``: interactive viewer of generated star systems.

Examples::

    python -m starsys.view                    # systems S1-0, S1-1, ... (Next / Previous)
    python -m starsys.view --id S7-1234       # start at a given system
    python -m starsys.view --real "tau Ceti" -n 20   # realisations conditioned on a real star
    python -m starsys.view --json out/systems.json   # systems exported by python -m starsys --json

Keys: PageDown / PageUp = Next / Previous.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m starsys.view", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1, help="master seed of the system sequence (default 1)")
    ap.add_argument("--id", help="start at this system ID, e.g. S1-42")
    ap.add_argument("--filter", default="all systems", help="initial Next/Previous filter (as in the window)")
    ap.add_argument("--real", help="view systems conditioned on this real star")
    ap.add_argument("-n", type=int, default=20, help="realisations of the real star (default 20)")
    ap.add_argument("--screenshot", help="render the window to this PNG and exit (for tests and docs)")
    ap.add_argument("--select", help="with --screenshot: select this body first")
    ap.add_argument("--size", default="1500x920", help="window size WxH")
    args = ap.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .app import BundleSource, SequenceSource, SystemWindow, dark_palette

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    dark_palette(app)
    start = 0
    if args.real:
        from ..generate import Systems
        from ..real import condition as cd
        from ..real.catalog import find
        rec = find(args.real)
        c = cd.condition(rec, n=args.n, seed=args.seed)
        source = BundleSource(Systems.from_conditioned(c), rec.name)
    else:
        source = SequenceSource(args.seed)
        if args.id:
            start = source.index_of(args.id)
            if start is None:
                ap.error(f"bad --id {args.id!r}")
    w = SystemWindow(source, start, args.filter)
    wd, ht = (int(v) for v in args.size.lower().split("x"))
    w.resize(wd, ht)
    w.show()
    if args.screenshot:
        def shot():
            if args.select:
                w.select(args.select)
            app.processEvents()
            w.grab().save(args.screenshot)
            app.quit()
        QTimer.singleShot(800, shot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
