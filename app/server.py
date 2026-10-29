"""
LogitGuard: Live GPU Logit Uncertainty & Hallucination Detector Server
Direct PyTorch lm_head extraction on Qwen-2.5-1.5B (RTX 5070 GPU).
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from logitguard.config import load_config, PROJECT_ROOT
from logitguard.model import load_model_and_tokenizer
from logitguard.pipeline import compare_methods_live, verify, rescore

app = FastAPI(title="LogitGuard Live Detector", version="2.0.0")

STATIC_DIR = PROJECT_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_CFG = load_config()
_CACHED_SAMPLES: Dict[str, Dict[str, Any]] = {}
_GPU_MODEL = None
_GPU_TOKENIZER = None
_GPU_LOCK = threading.Lock()
_INIT_TIME = 0.0
_START_TIME = time.perf_counter()


def initialize_model():
    """Load Qwen on GPU once on startup."""
    global _GPU_MODEL, _GPU_TOKENIZER, _INIT_TIME
    print(f"[LogitGuard] Loading model on GPU: {_CFG.get('model_name')}...")
    t0 = time.perf_counter()
    with _GPU_LOCK:
        _GPU_MODEL, _GPU_TOKENIZER = load_model_and_tokenizer(_CFG)
    _INIT_TIME = round(time.perf_counter() - t0, 2)
    print(f"[LogitGuard] Model ready on {_GPU_MODEL.device} in {_INIT_TIME}s")

    # Warm-up pass
    try:
        _ = compare_methods_live(
            _GPU_MODEL,
            _GPU_TOKENIZER,
            context="The sky is blue.",
            query="What color is the sky?",
            response="The sky is blue.",
            cfg=_CFG,
        )
        print("[LogitGuard] Warm-up successful.")
    except Exception as e:
        print(f"[LogitGuard] Warm-up warning: {e}")
    print("[LogitGuard] Ready to serve requests.")


def load_cached_samples():
    global _CACHED_SAMPLES
    cache_path = Path(_CFG["paths"]["demo_cache"])
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            sample_list = json.load(f)
            _CACHED_SAMPLES = {s["id"]: s for s in sample_list}


initialize_model()
load_cached_samples()


class RunRequest(BaseModel):
    context: Optional[str] = ""  # Optional reference context for RAG mode
    query: str = Field(..., min_length=1)
    response: Optional[str] = None  # If omitted or empty, Qwen generates it live on GPU
    gold_spans: Optional[List[Dict[str, Any]]] = None


class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    dtype: Optional[str] = "float16"


class RescoreRequest(BaseModel):
    sample_id: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None
    tau: Optional[float] = None
    variant: Optional[str] = None
    mask_mode: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def get_health():
    is_ready = _GPU_MODEL is not None and _GPU_TOKENIZER is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "ready": is_ready,
        "device": str(_GPU_MODEL.device) if _GPU_MODEL else "offline",
        "uptime_s": round(time.perf_counter() - _START_TIME, 1),
    }


@app.get("/status")
def get_status():
    import torch
    device = str(_GPU_MODEL.device) if _GPU_MODEL else "offline"
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    model_name = getattr(_GPU_MODEL, "name_or_path", None) or _CFG.get("model_name")
    dtype = _CFG.get("dtype", "float16")
    model_label = f"{model_name} ({dtype})" if dtype != "float16" else model_name
    return {
        "status": "ready" if _GPU_MODEL else "offline",
        "model": model_label,
        "base_model": model_name,
        "dtype": dtype,
        "device": device,
        "load_time_s": _INIT_TIME,
        "vram_mb": vram_mb,
    }


@app.get("/api/model")
def get_active_model():
    return get_status()


@app.post("/api/model/switch")
def switch_model(req: SwitchModelRequest):
    global _GPU_MODEL, _GPU_TOKENIZER, _INIT_TIME
    import gc
    import torch
    with _GPU_LOCK:
        print(f"[LogitGuard] Switching model to: {req.model_name} (dtype={req.dtype})...")
        t0 = time.perf_counter()
        _GPU_MODEL = None
        _GPU_TOKENIZER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        switch_cfg = dict(_CFG, model_name=req.model_name, dtype=req.dtype or "float16")
        try:
            _GPU_MODEL, _GPU_TOKENIZER = load_model_and_tokenizer(switch_cfg)
            _CFG["model_name"] = req.model_name
            _CFG["dtype"] = req.dtype or "float16"
            _INIT_TIME = round(time.perf_counter() - t0, 2)
            print(f"[LogitGuard] Model ready: {_CFG['model_name']} ({_CFG['dtype']}) on {_GPU_MODEL.device} in {_INIT_TIME}s")

            # Warm-up pass
            try:
                _ = compare_methods_live(
                    _GPU_MODEL,
                    _GPU_TOKENIZER,
                    context="The sky is blue.",
                    query="What color is the sky?",
                    response="The sky is blue.",
                    cfg=_CFG,
                )
            except Exception as we:
                print(f"[LogitGuard] Warm-up warning after switch: {we}")

        except Exception as e:
            # Fall back to restore previous model
            print(f"[LogitGuard] Error switching model: {e}. Restoring default...")
            _GPU_MODEL, _GPU_TOKENIZER = load_model_and_tokenizer(_CFG)
            raise HTTPException(status_code=500, detail=f"Failed to load model {req.model_name}: {e}")

    vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0
    return {
        "status": "ready",
        "model": _CFG["model_name"],
        "device": str(_GPU_MODEL.device) if _GPU_MODEL else "offline",
        "load_time_s": _INIT_TIME,
        "vram_mb": vram_mb,
    }


@app.post("/api/run")
def api_run(req: RunRequest):
    """
    Main Unified Pipeline:
    1. If response is empty: Qwen generates text live from (context, query).
    2. Runs single-pass GPU forward pass with lm_head hook to extract logits.
    3. Computes and compares:
       - LogitGuard (Ours): U = H / (1 + Delta) * w_entity
       - Raw Token Entropy (Fadeeva 2024): U = H
       - Min Concept Probability (Varshney 2023): U = 1 - P_max
       - Token Surprisal: U = -log P
    4. Returns text, per-token logit data (with Top-5 vocabulary distribution), and comparison.
    """
    import torch

    with _GPU_LOCK:
        if _GPU_MODEL is None or _GPU_TOKENIZER is None:
            raise HTTPException(status_code=503, detail="GPU model is not loaded.")

        context = (req.context or "").strip()
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Query cannot be empty or whitespace only.")

        response_text = (req.response or "").strip()
        gen_time_ms = 0.0

        # Step 1: If no response text supplied, generate with Qwen live
        if not response_text:
            if context:
                user_msg = f"Context:\n{context}\n\nInstruction:\n{query}"
                sys_msg = "You are a helpful and accurate assistant. Answer strictly based on the provided context."
            else:
                user_msg = query
                sys_msg = "You are a helpful and accurate assistant. Answer factually and truthfully."

            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ]
            prompt = _GPU_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = _GPU_TOKENIZER(prompt, return_tensors="pt").to(_GPU_MODEL.device)

            t_gen_start = time.perf_counter()
            with torch.no_grad():
                output_ids = _GPU_MODEL.generate(
                    **inputs,
                    max_new_tokens=140,
                    do_sample=False,
                    pad_token_id=_GPU_TOKENIZER.eos_token_id,
                )
            gen_time_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 1)
            gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            response_text = _GPU_TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

        # Step 2: Compare methods using live GPU forward hook
        try:
            results = compare_methods_live(
                _GPU_MODEL,
                _GPU_TOKENIZER,
                context=context,
                query=query,
                response=response_text,
                cfg=_CFG,
                gold_spans=req.gold_spans,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        results["generation_time_ms"] = gen_time_ms
        results["response"] = response_text

    return JSONResponse(content=results)


# Compatibility endpoints for integration tests
@app.post("/compare_live")
def api_compare_live(req: RunRequest):
    return api_run(req)


@app.post("/generate_live")
def api_generate_live(req: RunRequest):
    req.response = None
    return api_run(req)


@app.post("/verify")
def api_verify(req: RunRequest):
    run_cfg = dict(_CFG)
    with _GPU_LOCK:
        if _GPU_MODEL is None:
            raise HTTPException(status_code=503, detail="GPU model not loaded.")
        try:
            result = verify(_GPU_MODEL, _GPU_TOKENIZER, req.context or "", req.query, req.response or "", run_cfg)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
    return JSONResponse(content=result)


@app.get("/samples")
async def get_samples():
    if not _CACHED_SAMPLES:
        load_cached_samples()
    samples_meta = []
    for s_id, s in _CACHED_SAMPLES.items():
        samples_meta.append({
            "id": s_id,
            "task": s.get("task", "custom"),
            "has_gold": len(s.get("gold_spans", [])) > 0,
            "gold_count": len(s.get("gold_spans", [])),
            "response_preview": s.get("response", "")[:80] + "...",
            "context": s.get("context", ""),
            "query": s.get("query", ""),
            "response": s.get("response", ""),
        })
    return {"samples": samples_meta}


@app.post("/rescore")
async def api_rescore(req: RescoreRequest):
    if req.sample_id and req.sample_id in _CACHED_SAMPLES:
        tel = _CACHED_SAMPLES[req.sample_id]
    elif req.telemetry:
        tel = req.telemetry
    else:
        raise HTTPException(status_code=400, detail="Must provide sample_id or telemetry.")

    run_cfg = dict(_CFG)
    run_cfg["tau"] = req.tau if req.tau is not None else float(_CFG.get("tau", 0.58))
    run_cfg["variant"] = req.variant or _CFG.get("variant", "ccla")
    run_cfg["mask_mode"] = req.mask_mode or _CFG.get("mask_mode", "soft")

    result = rescore(tel, run_cfg)
    return JSONResponse(content=result)


@app.get("/benchmarks")
async def get_benchmarks():
    main_file = Path(_CFG["paths"]["main_table"])
    abl_file = Path(_CFG["paths"]["ablation_table"])
    lat_file = Path(_CFG["paths"]["latency_table"])

    def _rows(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        df = pd.read_csv(path)
        return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

    main_rows = _rows(main_file)
    abl_rows = _rows(abl_file)
    lat_rows = _rows(lat_file)

    return {
        "main_table": main_rows,
        "ablation_table": abl_rows,
        "latency_table": lat_rows,
    }
