# Local AI Commander

*[Read in Spanish / Leer en Español](#versión-en-español)*

Local MLOps command center for GGUF Large Language Models (LLMs) inference. Strictly designed to maximize performance on hardware with limited VRAM (RTX 4050 - 6GB) operating over WSL2 (Ubuntu).

## Architecture

* **Backend:** FastAPI + `llama-cpp-python`. Low-level memory management, forcing C++ pointer destruction and hot VRAM release.
* **Frontend:** Streamlit. Reactive interface with modular Workspaces, real-time CUDA telemetry, and SSE Streaming.
* **Memory (Workspaces):** State system based on the local file system. Allows strict separation between *Master Prompt* (Model's DNA) and *Dynamic Context* (Live memory intercepted and written by the LLM itself).

## System Requirements

* Windows 11 with WSL2 (Ubuntu 24.04).
* NVIDIA GPU (Optimized for 6GB VRAM RTX 40 series using Flash Attention).
* NVIDIA drivers installed on the Windows host.

## Installation (WSL2)

**1. Prepare base environment and CUDA compiler:**
```bash
sudo apt update
sudo apt install -y python3.12-venv python3-dev build-essential nvidia-cuda-toolkit
```

**2. Clone repository and create virtual environment:**
```bash
git clone https://github.com/MrDjiego/local-ai-commander.git
cd local-ai-commander
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**3. Install C++ engine forcing CUDA (RTX Offload):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**4. Install Dashboard dependencies:**
```bash
pip install -r requirements.txt
```

## Supported Models (GGUF)

Create the `models/` folder in the project root and download the weights. The current architecture is tuned for models in the Qwen 2.5 7B and Llama 3.1 8B families quantized in `Q4_K_M`.

## Execution

Launch the full system with the unified script:
```bash
chmod +x start.sh
./start.sh
```
The Dashboard will be available at `http://localhost:8501`.

---

# Versión en Español

Centro de comando MLOps local para inferencia de modelos de lenguaje (LLMs) GGUF. Diseñado estrictamente para maximizar el rendimiento en hardware con VRAM limitada (RTX 4050 - 6GB) operando sobre WSL2 (Ubuntu).

## Arquitectura

* **Backend:** FastAPI + `llama-cpp-python`. Gestión de memoria a bajo nivel, forzando la destrucción de punteros C++ y liberación de VRAM en caliente.
* **Frontend:** Streamlit. Interfaz reactiva con Workspaces modulares, telemetría de CUDA en tiempo real y Streaming SSE.
* **Memoria (Workspaces):** Sistema de estado basado en sistema de archivos local. Permite separación estricta entre *Master Prompt* (ADN del modelo) y *Contexto Dinámico* (Memoria viva interceptada y escrita por el propio LLM).

## Requisitos del Sistema

* Windows 11 con WSL2 (Ubuntu 24.04).
* GPU NVIDIA (Optimizado para 6GB VRAM de la serie RTX 40 con Flash Attention).
* Drivers de NVIDIA instalados en el host de Windows.

## Instalación (WSL2)

**1. Preparar el entorno base y compilador CUDA:**
```bash
sudo apt update
sudo apt install -y python3.12-venv python3-dev build-essential nvidia-cuda-toolkit
```

**2. Clonar el repositorio y crear entorno virtual:**
```bash
git clone https://github.com/MrDjiego/local-ai-commander.git
cd local-ai-commander
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**3. Instalar motor C++ forzando CUDA (RTX Offload):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**4. Instalar dependencias del Dashboard:**
```bash
pip install -r requirements.txt
```

## Modelos Soportados (GGUF)

Crea la carpeta `models/` en la raíz del proyecto y descarga los pesos. La arquitectura actual está afinada para modelos de la familia Qwen 2.5 7B y Llama 3.1 8B cuantizados en `Q4_K_M`.

## Ejecución

Lanza el sistema completo con el script unificado:
```bash
chmod +x start.sh
./start.sh
```
El Dashboard estará disponible en `http://localhost:8501`.