import gc
import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Local AI Commander - Backend")

llm: Llama | None = None
current_model_name: str | None = None

class LoadModelRequest(BaseModel):
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1

class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    min_p: float = 0.05
    repeat_penalty: float = 1.1

def get_gpu_telemetry():
    """Consulta directamente el driver NVIDIA en WSL2 para obtener métricas exactas."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(cmd, encoding="utf-8").strip()
        used, total, util, temp = [int(x.strip()) for x in output.split(",")]
        return {"vram_used": used, "vram_total": total, "gpu_util": util, "temp": temp}
    except Exception:
        return {"vram_used": 0, "vram_total": 6144, "gpu_util": 0, "temp": 0}

@app.get("/status")
def get_status():
    return {
        "loaded": llm is not None,
        "model": current_model_name,
        "gpu": get_gpu_telemetry()
    }

@app.post("/load_model")
def load_model(req: LoadModelRequest):
    global llm, current_model_name
    
    if llm is not None:
        return {"status": "already_loaded", "model": current_model_name}
    
    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"No existe: {req.model_path}")
    
    try:
        llm = Llama(
            model_path=req.model_path,
            n_gpu_layers=req.n_gpu_layers,
            n_ctx=req.n_ctx,
            flash_attn=True,
            verbose=False
        )
        current_model_name = os.path.basename(req.model_path)
        return {"status": "loaded", "model": current_model_name}
    except Exception as e:
        llm = None
        current_model_name = None
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model")
def unload_model():
    global llm, current_model_name
    if llm is None:
        return {"status": "unloaded"}
    
    try:
        llm.close()
        del llm
        llm = None
        current_model_name = None
        gc.collect()
        return {"status": "unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    global llm
    if llm is None:
        raise HTTPException(status_code=400, detail="Modelo apagado.")
    
    def token_generator():
        stream = llm.create_chat_completion(
            messages=req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            min_p=req.min_p,
            repeat_penalty=req.repeat_penalty,
            stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                # Enviamos cada token en formato JSON por línea
                yield json.dumps({"token": delta["content"]}) + "\n"

    return StreamingResponse(token_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)