#!/usr/bin/env bash
# Instala (o actualiza) el temporizador diario de equitydash como unidad systemd de usuario.
# Idempotente: volver a ejecutarlo reescribe las unidades y recarga.
#
# Para que corra sin sesión iniciada hace falta "linger": loginctl enable-linger "$USER".
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "No existe $REPO/.venv/bin/python: crea el entorno virtual primero." >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
sed "s|@REPO@|$REPO|g" "$REPO/deploy/equitydash-daily.service.in" \
    > "$UNIT_DIR/equitydash-daily.service"
cp "$REPO/deploy/equitydash-daily.timer" "$UNIT_DIR/equitydash-daily.timer"

systemctl --user daemon-reload
systemctl --user enable --now equitydash-daily.timer

if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    echo "Aviso: linger desactivado; el temporizador solo corre con la sesión iniciada." >&2
    echo "       Actívalo con: loginctl enable-linger $USER" >&2
fi
systemctl --user list-timers equitydash-daily.timer --all
