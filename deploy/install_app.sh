#!/usr/bin/env bash
# Instala (o actualiza) el panel local de equitydash como servicio systemd de usuario: arranca
# con el equipo y se reinicia si se cae. Idempotente: volver a ejecutarlo reescribe la unidad
# y reinicia el panel (úsalo también tras actualizar el código).
#
# Para que arranque sin sesión iniciada hace falta "linger": loginctl enable-linger "$USER".
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "No existe $REPO/.venv/bin/python: crea el entorno virtual primero." >&2
    exit 1
fi
if ! "$REPO/.venv/bin/python" -c "import streamlit" 2>/dev/null; then
    echo "Falta Streamlit en el entorno: pip install -e \".[app]\"." >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
sed "s|@REPO@|$REPO|g" "$REPO/deploy/equitydash-app.service.in" \
    > "$UNIT_DIR/equitydash-app.service"

systemctl --user daemon-reload
systemctl --user enable equitydash-app.service
systemctl --user restart equitydash-app.service

if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    echo "Aviso: linger desactivado; el panel solo corre con la sesión iniciada." >&2
    echo "       Actívalo con: loginctl enable-linger $USER" >&2
fi
systemctl --user --no-pager status equitydash-app.service | head -5
echo "Panel en http://localhost:8502"
