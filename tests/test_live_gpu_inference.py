import time
import torch
from logitguard.config import load_config
from logitguard.model import load_model_and_tokenizer
from logitguard.pipeline import verify

print("[Live GPU Test] Initializing configuration...")
cfg = load_config()

print(f"[Live GPU Test] Loading model '{cfg['model_name']}' on GPU ({torch.cuda.get_device_name(0)})...")
t0 = time.perf_counter()
model, tokenizer = load_model_and_tokenizer(cfg)
load_time = time.perf_counter() - t0
print(f"[Live GPU Test] Model loaded in {load_time:.2f}s on {model.device} (dtype: {model.dtype})")

context = '{"name": "Bistro Luna", "city": "Austin", "WiFi": "no", "stars": 4.5}'
query = "Describe the restaurant based strictly on the JSON attributes."
response = "Bistro Luna is a 4.5-star restaurant in Austin that offers complimentary high-speed Wi-Fi."

print("\n[Live GPU Test] Executing single-pass gray-box verification...")
result = verify(model, tokenizer, context, query, response, cfg)

print(f"[Live GPU Test] Verification complete!")
print(f"  * Engine: {result['engine']}")
print(f"  * Model Forward: {result['timing_ms']['model']} ms")
print(f"  * LogitGuard Overhead: {result['timing_ms']['logitguard']} ms")
print(f"  * Total Execution: {result['timing_ms']['total']} ms")
print(f"  * Peak VRAM Delta: {result['peak_vram_delta_mb']} MB")

print("\n[Live GPU Test] Candidate Spans Evaluated:")
for s in result["spans"]:
    status = "[FLAGGED: HALLUCINATION]" if s["flagged"] else "[GROUNDED: SUPPORTED]"
    print(f"  - [{s['char_start']}:{s['char_end']}] \"{s['text']}\" -> Score: {s['score']:.4f} (Tau: {result['tau']}) -> {status}")

print("\n[Live GPU Test] Per-Token Telemetry on Key Words:")
for t in result["tokens"]:
    if t["is_anchor"] or t["H"] > 1.0:
        top_cand = t["top5"][0]["token"] if t["top5"] else ""
        text_safe = t['text'].encode('ascii', errors='replace').decode('ascii')
        cand_safe = top_cand.encode('ascii', errors='replace').decode('ascii')
        print(f"  - Token: '{text_safe}' | H: {t['H']:.2f} | Delta: {t['delta']:.2f} | w: {t['w']} | U: {t['U']:.4f} | Top-1: '{cand_safe}'")
