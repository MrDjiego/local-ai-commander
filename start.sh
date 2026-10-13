#!/bin/bash
source ~/local_ia/venv/bin/activate

echo "[+] Iniciando Backend FastAPI (Puerto 8000)..."
python ~/local_ia/main.py &
BACKEND_PID=$!

# Función para matar el backend y limpiar VRAM cuando cierres con Ctrl+C
cleanup() {
    echo -e "\n[-] Apagando servicios y liberando VRAM de la RTX 4050..."
    kill $BACKEND_PID
    exit 0
}
trap cleanup SIGINT SIGTERM

sleep 2
echo "[+] Iniciando Dashboard Streamlit (Puerto 8501)..."
streamlit run ~/local_ia/app.py --server.headless true