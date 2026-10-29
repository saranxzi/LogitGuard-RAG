"""
Configuration Manager for LogitGuard-RAG
Loads config.yaml, resolves relative paths against the project root,
and validates parameters against allowable schema values.
"""

import copy
from pathlib import Path
from typing import Dict, Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "fallback_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "dtype": "float16",
    "attn_implementation": "sdpa",
    "max_tokens": 3072,
    "chunk_size": 64,
    "seed": 42,
    "variant": "ccla",
    "agg": "hybrid",
    "mask_mode": "soft",
    "weights": {"entity": 1.0, "other": 0.1},
    "max_span_gap": 2,
    "max_span_tokens": 32,
    "tau": 0.58,
    "paths": {
        "raw_dir": "data/raw",
        "clean_data": "data/ragtruth_clean.jsonl",
        "perturb_data": "data/perturbations.jsonl",
        "demo_cache": "data/cached_demo_samples.json",
        "results_dir": "results",
        "tau_file": "results/tau.json",
        "main_table": "results/main_table.csv",
        "ablation_table": "results/ablation_table.csv",
        "latency_table": "results/latency.csv",
    },
}

VALID_VARIANTS = {"core", "ccla", "contrastive", "entropy", "margin", "min_prob", "surprisal", "surprisal_margin"}
VALID_MASK_MODES = {"soft", "hard", "uniform"}


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Loads, merges, and validates configuration.
    Resolves relative path values to absolute Paths based on PROJECT_ROOT.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
            for k, v in user_cfg.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v

    # Validation
    if cfg["variant"] not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{cfg['variant']}'. Must be one of {VALID_VARIANTS}")
    if cfg["mask_mode"] not in VALID_MASK_MODES:
        raise ValueError(f"Invalid mask_mode '{cfg['mask_mode']}'. Must be one of {VALID_MASK_MODES}")
    if cfg["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")

    # Resolve paths to absolute paths
    paths = dict(cfg.get("paths", {}))
    resolved_paths = {}
    for p_key, p_val in paths.items():
        p_obj = Path(p_val)
        if not p_obj.is_absolute():
            resolved_paths[p_key] = str(PROJECT_ROOT / p_obj)
        else:
            resolved_paths[p_key] = str(p_obj)
    cfg["paths"] = resolved_paths

    return cfg
