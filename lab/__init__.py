"""Laboratorio de reglas: probar hipótesis sobre los datos del panel, con disciplina.

Uso mínimo (ver ``experimentos/plantilla.py``)::

    from lab import Lab, test, summary

    lab = Lab()
    for r in test(lab, "knife", max_return=-0.25):     # una regla del catálogo, otro umbral
        print(r)
    print(summary())                                   # todo lo probado, con BH global

Las piezas:

- ``lab.data.Lab``        los datos, point-in-time y sin red (§9.4, §9.1, §9.5);
- ``lab.signals``         el catálogo de reglas (``catalogue()``), las del panel y nuevas;
- ``lab.evaluate``        las tres pruebas (``timing``, ``cross_section``, ``events``), el
                          simulador de estrategia (``strategy``) y el de aportes con freno
                          (``contributions``);
- ``lab.registry``        el registro de todo lo probado y la corrección por pruebas
                          múltiples sobre el total (§9.7).

Reglas de la casa, que el laboratorio aplica sin que tengas que acordarte:

1. Nada se ve antes de publicarse: la macro por fecha de publicación, los fundamentales por
   fecha de presentación, la entrada al cierre de la sesión siguiente.
2. Se compara contra una base (el resto del día, los días normales, la misma empresa otro
   día), nunca contra cero.
3. Cada prueba queda registrada; ``summary()`` corrige por **todas**.
4. ``lab.holdout_from`` (settings) reserva el final del periodo: las pruebas no lo ven hasta
   que lo pidas con ``holdout=True``, una vez, con la regla ya fijada.
5. Una curva de ``strategy`` que bate a SPY no valida nada: primero las pruebas.

⚠️ CLAUDE.md §12 excluye los *backtests de trading* del panel porque invitan a operar a
menudo. Esto es el marco de validación de la fase 4 abierto a tus hipótesis: resolución
diaria, horizontes de meses y ninguna salida hacia una orden.
"""

from __future__ import annotations

from typing import Any, Sequence

from lab import evaluate, registry, signals
from lab.data import Lab
from lab.registry import summary
from lab.signals import RULES, catalogue

__all__ = ["Lab", "test", "summary", "catalogue", "RULES", "evaluate", "signals",
           "registry"]


def test(lab: Lab, name: str, *, horizons: Sequence[int] | None = None,
         holdout: bool = False, record: bool = True, **params: Any):
    """Run the catalogue rule ``name`` with ``params`` overriding its defaults, through the
    test its kind calls for. Returns the list of ``evaluate.Result`` (the per-date or
    per-event table too, for cross-section and events: ``(results, table)``)."""
    spec = RULES[name]
    output, merged = signals.build(lab, name, **params)
    common = {"rule": name, "holdout": holdout, "params": merged, "record": record,
              "expected": spec.expected}
    if spec.kind == "timing":
        return evaluate.timing(lab, output, horizons=horizons or (30, 90, 180), **common)
    if spec.kind == "cross":
        return evaluate.cross_section(lab, output, horizons=horizons or (90, 180),
                                      mode=spec.mode, **common)
    return evaluate.events(lab, output, horizons=horizons or (30, 90, 180), **common)
