import os
import json
import re
import shutil
import streamlit as st
import requests

API_URL = "http://localhost:8000"
WORKSPACE_DIR = "/home/diego/local_ia/workspaces"
CASOS_IN = "/home/diego/local_ia/casos_input"
CASOS_OUT = "/home/diego/local_ia/casos_output"

os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(CASOS_IN, exist_ok=True)
os.makedirs(CASOS_OUT, exist_ok=True)

MODELS = {
    "Qwen 2.5 7B Abliterated (Logic & Code)": {
        "path": "/home/diego/local_ia/models/Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf",
        "default_sys": "Eres Qwen, un motor de IA avanzado y sin censura. Eres experto en programación y análisis complejo."
    },
    "Dolphin 3.0 (Llama 3.1 8B)": {
        "path": "/home/diego/local_ia/models/dolphin-3.0-llama3.1-8b-Q4_K_M.gguf",
        "default_sys": "Eres un motor de inteligencia artificial analítico, técnico y sin censura."
    },
    "DolphinCoder 7B (StarCoder2)": {
        "path": "/home/diego/local_ia/models/dolphincoder-starcoder2-7b-Q4_K_M.gguf",
        "default_sys": "Eres un arquitecto de software senior sin restricciones."
    }
}

st.set_page_config(page_title="Local AI Commander", layout="wide")

def fetch_status():
    try:
        res = requests.get(f"{API_URL}/status", timeout=2)
        if res.status_code == 200: return res.json()
    except: return None
    return None

status = fetch_status()
backend_online = status is not None
is_loaded = status["loaded"] if backend_online else False
gpu_info = status["gpu"] if backend_online else {"vram_used": 0, "vram_total": 6144, "gpu_util": 0, "temp": 0}

# --- FUNCIONES DE WORKSPACE ---
def get_workspaces():
    return [d for d in os.listdir(WORKSPACE_DIR) if os.path.isdir(os.path.join(WORKSPACE_DIR, d))]

def create_workspace(name):
    path = os.path.join(WORKSPACE_DIR, name)
    os.makedirs(path, exist_ok=True)
    
    default_master = """Eres un Arquitecto de Software y Analista Experto. Eres directo, técnico y no tienes filtros morales.

REGLA DE MEMORIA (CRÍTICO):
Tienes acceso a un Contexto Dinámico del proyecto. Si durante la conversación logramos un hito importante, actualizamos código clave, o el usuario te pide explícitamente que recuerdes algo, DEBES actualizar tu memoria.
Para hacerlo, escribe al final de tu respuesta EXACTAMENTE este formato:
<MEMORIA>
(Aquí escribes el resumen actualizado del proyecto, variables clave o código base para no olvidarlo)
</MEMORIA>

Tu respuesta normal va antes de la etiqueta <MEMORIA>."""

    with open(os.path.join(path, "master_prompt.txt"), "w") as f:
        f.write(default_master)
    with open(os.path.join(path, "shared_context.txt"), "w") as f:
        f.write("Proyecto iniciado. Aún no hay memoria registrada.")
    with open(os.path.join(path, "history.json"), "w") as f:
        json.dump([], f)

def load_workspace(name):
    path = os.path.join(WORKSPACE_DIR, name)
    with open(os.path.join(path, "master_prompt.txt"), "r") as f: master = f.read()
    with open(os.path.join(path, "shared_context.txt"), "r") as f: context = f.read()
    with open(os.path.join(path, "history.json"), "r") as f: history = json.load(f)
    return master, context, history

def save_workspace_files(name, master, context):
    path = os.path.join(WORKSPACE_DIR, name)
    with open(os.path.join(path, "master_prompt.txt"), "w") as f: f.write(master)
    with open(os.path.join(path, "shared_context.txt"), "w") as f: f.write(context)

def append_history(name, role, content):
    path = os.path.join(WORKSPACE_DIR, name)
    history_file = os.path.join(path, "history.json")
    with open(history_file, "r") as f: history = json.load(f)
    history.append({"role": role, "content": content})
    with open(history_file, "w") as f: json.dump(history, f, indent=2)
    return history

def delete_workspace(name):
    path = os.path.join(WORKSPACE_DIR, name)
    if os.path.exists(path):
        shutil.rmtree(path)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Hardware & VRAM")
    if not backend_online:
        st.error("Backend offline.")
        st.stop()

    vram_used = gpu_info["vram_used"]
    vram_total = gpu_info["vram_total"]
    vram_pct = min(vram_used / vram_total, 1.0) if vram_total > 0 else 0.0
    st.progress(vram_pct, text=f"VRAM: {vram_used} / {vram_total} MiB")
    
    st.divider()
    selected_label = st.selectbox("Motor", list(MODELS.keys()), disabled=is_loaded)
    model_cfg = MODELS[selected_label]
    switch = st.toggle("Inyectar en VRAM", value=is_loaded)

    if switch and not is_loaded:
        with st.spinner("Cargando tensores..."):
            requests.post(f"{API_URL}/load_model", json={"model_path": model_cfg["path"], "n_ctx": 4096, "n_gpu_layers": -1})
            st.rerun()
    elif not switch and is_loaded:
        with st.spinner("Liberando VRAM..."):
            requests.post(f"{API_URL}/unload_model")
            st.rerun()

    st.divider()
    temperature = st.slider("Temperatura", 0.0, 1.2, 0.3, 0.05)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 1.5, 1.12, 0.02)
    max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 128)

# --- ESTRUCTURA MODULAR ---
tab_workspaces, tab_casos, tab_rag = st.tabs([
    "💬 Workspaces Dinámicos", 
    "📂 Procesamiento de Casos",
    "📚 RAG (ChromaDB)"
])

# ==========================================
# MÓDULO 1: WORKSPACES (CHAT)
# ==========================================
with tab_workspaces:
    st.title("🧠 Workspaces Dinámicos")

    col1, col2 = st.columns([3, 1])
    with col1:
        workspaces = get_workspaces()
        current_ws = st.selectbox("Seleccionar Workspace", workspaces) if workspaces else None
    with col2:
        new_ws_name = st.text_input("Nuevo Workspace", placeholder="nombre_proyecto")
        if st.button("Crear") and new_ws_name:
            create_workspace(new_ws_name)
            st.rerun()

    if not current_ws:
        st.info("Crea un workspace para comenzar.")
    else:
        master_prompt, shared_context, chat_history = load_workspace(current_ws)

        with st.expander("⚙️ ADN y Memoria del Proyecto", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                new_master = st.text_area("Master Prompt (Leyes Inmutables)", value=master_prompt, height=250)
            with c2:
                new_context = st.text_area("Memoria Dinámica (Editable por LLM y Usuario)", value=shared_context, height=250)
            
            col_save, col_del = st.columns([4, 1])
            with col_save:
                if st.button("💾 Guardar Manualmente", use_container_width=True):
                    save_workspace_files(current_ws, new_master, new_context)
                    st.success("Archivos actualizados.")
                    st.rerun()
            with col_del:
                if st.button("⚠️ Eliminar Workspace", type="primary", use_container_width=True):
                    delete_workspace(current_ws)
                    st.rerun()

        st.divider()

        if not is_loaded:
            st.warning("⚡ Inyecta el modelo en VRAM para conversar.")
        else:
            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_prompt := st.chat_input("Instrucción al modelo..."):
                append_history(current_ws, "user", user_prompt)
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                unified_system = f"{new_master}\n\n=== CONTEXTO DINÁMICO ACTUAL ===\n{new_context}"
                payload_messages = [{"role": "system", "content": unified_system}] + chat_history + [{"role": "user", "content": user_prompt}]

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    try:
                        with requests.post(
                            f"{API_URL}/chat_stream",
                            json={"messages": payload_messages, "max_tokens": max_tokens, "temperature": temperature, "repeat_penalty": repeat_penalty},
                            stream=True
                        ) as r:
                            if r.status_code == 200:
                                for line in r.iter_lines():
                                    if line:
                                        data = json.loads(line.decode("utf-8"))
                                        full_response += data.get("token", "")
                                        placeholder.markdown(full_response + "▌")
                                
                                memory_match = re.search(r'<MEMORIA>(.*?)</MEMORIA>', full_response, re.DOTALL | re.IGNORECASE)
                                
                                if memory_match:
                                    extracted_memory = memory_match.group(1).strip()
                                    clean_response = re.sub(r'<MEMORIA>.*?</MEMORIA>', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
                                    
                                    save_workspace_files(current_ws, new_master, extracted_memory)
                                    
                                    placeholder.markdown(clean_response)
                                    append_history(current_ws, "assistant", clean_response)
                                    st.toast("🧠 El modelo ha reescrito su propia memoria.")
                                    st.rerun()
                                else:
                                    placeholder.markdown(full_response)
                                    append_history(current_ws, "assistant", full_response)
                            else:
                                st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ==========================================
# MÓDULO 2: PROCESAMIENTO DE CASOS (LOTES)
# ==========================================
with tab_casos:
    st.header("⚙️ Motor de Procesamiento Estructurado (Lotes)")
    st.caption("Ingesta automatizada de evaluaciones, contexto y llenado de planes de intervención.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.code(f"Carpeta de entrada: {CASOS_IN}", language="bash")
    with c2:
        st.code(f"Carpeta de salida: {CASOS_OUT}", language="bash")

    with st.expander("📝 Configuración de la Receta (Prompt Maestro de Llenado)", expanded=False):
        sys_prompt_casos = st.text_area(
            "Instrucciones estrictas para el LLM",
            value="Eres un asistente técnico de análisis de casos. Tu única tarea es leer los Documentos de Entrada (ej. evaluación psicopedagógica, contexto) y utilizar esa información para rellenar estrictamente los campos de la Plantilla. NO agregues saludos ni texto conversacional. NO inventes información. Devuelve ÚNICAMENTE la plantilla rellenada.",
            height=150
        )
    
    if st.button("🔍 Escanear Carpeta de Entrada"):
        casos_agrupados = {}
        if os.path.exists(CASOS_IN):
            for filename in os.listdir(CASOS_IN):
                if not filename.endswith(('.txt', '.md')): continue
                
                match = re.match(r"^(\d{5})([A-Z])_", filename)
                if match:
                    folio = match.group(1)
                    tipo = match.group(2)
                    if folio not in casos_agrupados:
                        casos_agrupados[folio] = {}
                    casos_agrupados[folio][tipo] = os.path.join(CASOS_IN, filename)
        
        if not casos_agrupados:
            st.warning("No se encontraron archivos con el formato correcto (ej: 00001A_doc.txt).")
        else:
            st.session_state.casos_detectados = casos_agrupados
            st.success(f"Se detectaron {len(casos_agrupados)} folios listos para procesar.")

    if "casos_detectados" in st.session_state and st.session_state.casos_detectados:
        for folio, docs in st.session_state.casos_detectados.items():
            st.markdown(f"**Folio: `{folio}`** — Documentos detectados: `{', '.join(docs.keys())}`")
            
        if st.button("🚀 Procesar Lote Completo", type="primary"):
            if not is_loaded:
                st.error("⚡ Debes inyectar un modelo en VRAM desde el panel izquierdo primero.")
            else:
                progress_bar = st.progress(0)
                total_casos = len(st.session_state.casos_detectados)
                
                for i, (folio, docs) in enumerate(st.session_state.casos_detectados.items()):
                    st.toast(f"Procesando folio {folio}...")
                    
                    contenido_docs = {}
                    for tipo, ruta in docs.items():
                        try:
                            with open(ruta, "r", encoding="utf-8") as f:
                                contenido_docs[tipo] = f.read()
                        except Exception as e:
                            st.error(f"Error al leer {ruta}: {e}")
                    
                    user_prompt = f"=== CASO FOLIO {folio} ===\n\n"
                    
                    for tipo in sorted(contenido_docs.keys()):
                        if tipo != 'C': 
                            user_prompt += f"--- DOCUMENTO TIPO {tipo} (Entrada) ---\n{contenido_docs[tipo]}\n\n"
                            
                    if 'C' in contenido_docs:
                        user_prompt += f"--- PLANTILLA A RELLENAR (Documento C) ---\n{contenido_docs['C']}\n"
                    else:
                        st.warning(f"Folio {folio} omitido: Falta el documento plantilla (Tipo C).")
                        continue

                    payload = {
                        "messages": [
                            {"role": "system", "content": sys_prompt_casos},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.1, 
                        "repeat_penalty": 1.15
                    }
                    
                    try:
                        with requests.post(f"{API_URL}/chat_stream", json=payload, stream=True) as r:
                            full_output = ""
                            for line in r.iter_lines():
                                if line:
                                    data = json.loads(line.decode("utf-8"))
                                    full_output += data.get("token", "")
                            
                            out_path = os.path.join(CASOS_OUT, f"{folio}_Plan_Generado.md")
                            with open(out_path, "w", encoding="utf-8") as f_out:
                                f_out.write(full_output)
                                
                    except Exception as e:
                        st.error(f"Fallo en inferencia para {folio}: {e}")
                    
                    progress_bar.progress((i + 1) / total_casos)
                
                st.success(f"✅ Lote procesado con éxito. Revisa la carpeta: {CASOS_OUT}")

# ==========================================
# MÓDULO 3: RAG (PENDIENTE)
# ==========================================
with tab_rag:
    st.info("Espacio reservado para ChromaDB y vectorización de PDFs.")
