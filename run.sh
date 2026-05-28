#!/bin/bash

VENV_DIR=".env"
REQUIREMENTS="requirements.txt"
MAIN="Red/camaraNumeros.py"

cd "$(dirname "$0")"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv "$VENV_DIR"
    echo "Instalando dependencias..."
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"
fi

echo "Iniciando programa..."
"$VENV_DIR/bin/python" "$MAIN"

if [ $? -ne 0 ]; then
    echo "Error al iniciar. Reinstalando dependencias..."
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"
    echo "Reintentando..."
    "$VENV_DIR/bin/python" "$MAIN"
fi
