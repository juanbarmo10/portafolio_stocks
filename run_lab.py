"""Laboratorio de reglas desde la línea de comandos (``lab/``).

    python run_lab.py reglas                               # el catálogo
    python run_lab.py probar knife                         # una regla con sus valores
    python run_lab.py probar knife --param max_return=-0.3 --horizontes 90 180
    python run_lab.py probar regime_vote --param component=curve
    python run_lab.py bateria                              # todas, con sus valores por defecto
    python run_lab.py estrategia momentum --top 25         # simulación, NO validación
    python run_lab.py aportes brake --param variant=V2_tendencia
    python run_lab.py registro                             # todo lo probado, BH global
    python run_lab.py probar knife --holdout               # el periodo reservado: una vez

Para experimentos propios en Python: ``experimentos/plantilla.py``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd
import yaml

from core.logging_setup import configure_logging


def _params(pairs: list[str] | None) -> dict[str, Any]:
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _show(results) -> None:
    from lab.evaluate import table  # noqa: PLC0415

    frame = table(results)
    if frame.empty:
        print("Sin observaciones.")
        return
    shown = frame[["rule", "kind", "horizon", "n", "mean", "hit", "p", "half_1", "half_2",
                   "note"]].copy()
    for c in ("mean", "half_1", "half_2"):
        shown[c] = shown[c].map(lambda x: "—" if x is None or pd.isna(x) else f"{x*100:+.2f} %")
    shown["hit"] = shown["hit"].map(lambda x: "—" if x is None or pd.isna(x) else f"{x:.0%}")
    shown["p"] = shown["p"].map(lambda x: "—" if x is None or pd.isna(x) else f"{x:.3f}")
    print(shown.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Laboratorio de reglas (lab/).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reglas", help="el catálogo de reglas")
    one = sub.add_parser("probar", help="probar una regla")
    one.add_argument("regla")
    one.add_argument("--param", action="append", help="clave=valor (varias veces)")
    one.add_argument("--horizontes", nargs="+", type=int)
    one.add_argument("--holdout", action="store_true", help="usar el periodo reservado")
    all_ = sub.add_parser("bateria", help="todas las reglas con sus valores por defecto")
    all_.add_argument("--holdout", action="store_true")
    strat = sub.add_parser("estrategia", help="simular una cartera con una regla transversal")
    strat.add_argument("regla")
    strat.add_argument("--param", action="append")
    strat.add_argument("--top", type=int, default=25, help="acciones con mejor puntuación")
    strat.add_argument("--rebalanceo", type=int, default=91, help="días entre rebalanceos")
    strat.add_argument("--coste", type=float, default=10.0, help="coste por operación, pb")
    strat.add_argument("--holdout", action="store_true")
    dca = sub.add_parser("aportes", help="aportes mensuales con una regla temporal de freno")
    dca.add_argument("regla")
    dca.add_argument("--param", action="append")
    dca.add_argument("--holdout", action="store_true")
    sub.add_parser("registro", help="todo lo probado, con BH sobre el total")
    args = parser.parse_args(argv)
    configure_logging()

    from lab import RULES, Lab, catalogue, evaluate, signals, summary, test  # noqa: PLC0415

    if args.command == "reglas":
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(catalogue().to_string(index=False))
        return 0
    if args.command == "registro":
        frame = summary()
        if frame.empty:
            print("El registro está vacío: todavía no has probado nada.")
            return 0
        in_sample = frame[~frame["holdout"].astype(bool)]
        print(f"{len(in_sample)} pruebas distintas en la muestra; "
              f"{int(in_sample['significant'].fillna(False).sum())} significativas tras BH "
              f"sobre TODAS. Holdout usado {int(frame['holdout'].astype(bool).sum())} veces.")
        with pd.option_context("display.width", 220, "display.max_colwidth", 40):
            print(frame.drop(columns=["params"]).to_string(index=False))
        return 0

    lab = Lab()
    if args.command == "probar":
        if args.regla not in RULES:
            parser.error(f"regla desconocida {args.regla!r}: `python run_lab.py reglas`")
        out = test(lab, args.regla, horizons=args.horizontes, holdout=args.holdout,
                   **_params(args.param))
        _show(out[0] if isinstance(out, tuple) else out)
        return 0
    if args.command == "bateria":
        results = []
        for name in RULES:
            try:
                out = test(lab, name, holdout=args.holdout)
            except FileNotFoundError as exc:          # a cache that was never filled
                print(f"{name}: saltada — {exc}")
                continue
            results += out[0] if isinstance(out, tuple) else out
        _show(results)
        print("\nBH sobre todo el registro: `python run_lab.py registro`.")
        return 0
    if args.command == "estrategia":
        spec = RULES.get(args.regla)
        if spec is None or spec.kind != "cross":
            parser.error("la estrategia necesita una regla transversal (tipo cross)")
        scores, _ = signals.build(lab, args.regla, **_params(args.param))
        sign = 1 if spec.expected == "higher" else -1

        def select(day, members):
            s = scores(day, members)
            if s is None or s.empty:
                return []
            if s.dtype == bool:          # a flag: every flagged name ("lower": the others)
                return list(s.index[s if sign > 0 else ~s])
            return list((s * sign).sort_values(ascending=False).head(args.top).index)
        bt = evaluate.strategy(lab, select, rebalance_days=args.rebalanceo,
                               cost_bps=args.coste, holdout=args.holdout)
        print(bt)
        print("⚠️ Simulación, no validación: sin la prueba de la regla y el holdout, una curva "
              "que bate a SPY no dice nada. Sesgo de supervivencia a favor (§2.24).")
        return 0
    if args.command == "aportes":
        spec = RULES.get(args.regla)
        if spec is None or spec.kind != "timing":
            parser.error("los aportes necesitan una regla temporal (tipo timing)")
        brake, _ = signals.build(lab, args.regla, **_params(args.param))
        out = evaluate.contributions(lab, brake, holdout=args.holdout)
        a, b = out["con_freno"], out["siempre"]
        print(f"Con freno: riqueza final {a.final_wealth:.3f} por unidad aportada, "
              f"{a.months_deferred} aportes aplazados. Siempre: {b.final_wealth:.3f}. "
              f"Diferencia {out['diferencia'] * 100:+.2f} %.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
