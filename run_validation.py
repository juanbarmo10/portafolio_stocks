"""Validation report entry point (CLAUDE.md section 8, phase 4).

Runs the pre-registered battery of :mod:`validation.backtest` once and prints the report —
including, and first, what does not work. Re-running gives the same numbers: the shuffles
are seeded and the data is point-in-time.

Usage:
    python run_validation.py                    # print the report
    python run_validation.py --out informe.md   # also write it to a file
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from core.config import load_settings
from core.logging_setup import configure_logging, get_logger
from db.database import open_connection, read_observations, read_table
from transform import regime as rg
from validation.backtest import discrimination, report, run_battery

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="equitydash signal validation")
    parser.add_argument("--out", help="Also write the Markdown report to this file.")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())
    level2 = settings.raw["panel"]["level2"]
    cfg = settings.raw.get("validation", {})

    conn = open_connection(settings.db_path)
    try:
        built = rg.build(
            read_observations(conn, source="fred"),
            read_observations(conn, series_ids=[f"{t}:close_raw"
                                                for t in rg.regime_tickers(level2)]),
            read_table(conn, "corporate_actions"), level2,
        )
    finally:
        conn.close()
    if built is None:
        log.error("No data for the regime light: run `python run_ingest.py --only fred prices`.")
        return 1

    battery = run_battery(built, cfg)
    judged = built.frame.index[built.frame["verdict"] != rg.INSUFFICIENT]
    verdict_start = judged[0].date().isoformat() if len(judged) else None
    key = str(cfg.get("out_of_sample_component", "equal_weight"))
    in_sample = discrimination(built, key, start=verdict_start)
    oos = [discrimination(built, key, end=verdict_start)]
    battery.notes += [
        f"Dentro de muestra = la ventana con veredicto (desde {verdict_start}), la que miró "
        "la fase 3. Fuera de muestra = todo lo anterior en que el componente existe.",
        "La hipótesis de NFCI no se puede probar fuera de muestra: su historia point-in-time "
        "empieza en 2011 y lo anterior son valores revisados (look-ahead, §9.4).",
    ]
    text = report(battery, oos, in_sample, date.today().isoformat())
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
