"""
vllm/benchmark_throughput_local.py — Local throughput benchmark for DPO checkpoint.

vLLM requires Linux/WSL2 and is not available natively on Windows.
This script benchmarks the DPO checkpoint (Qwen2.5-1.5B + LoRA) using
HuggingFace generate() on the RTX 4090, measuring:
  - TTFT  (time-to-first-token, approximated as generation start latency)
  - TPS   (tokens per second)
  - Total throughput across 10 sequential queries

Compare against Groq baseline (llama-3.3-70b-versatile) to document tradeoff.

Run:  C:\\Users\\suchi\\anaconda3\\python.exe vllm/benchmark_throughput_local.py
"""

import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DPO_DIR = ROOT / "dpo" / "dpo_checkpoint"

QUERIES = [
    "What anti-dumping duty has India imposed on Chinese hot-rolled steel coils?",
    "Explain India's safeguard measures on flat steel imports.",
    "What is India's position on EU CBAM for steel exporters?",
    "How does the India-UAE CEPA affect steel exports?",
    "What coking coal supply risks does India face from Australia?",
    "What are BIS quality control order requirements for steel imports?",
    "How has India's steel import volume changed from China in 2024?",
    "What is the World Steel Association forecast for Indian steel demand?",
    "Explain the PLI scheme benefits for specialty steel producers in India.",
    "What anti-dumping duties does India have on seamless tubes from China?",
]

SYSTEM_PROMPT = (
    "You are a steel trade policy analyst. Answer concisely in 2-3 sentences "
    "based on your knowledge of Indian steel trade policy."
)


def run_groq_baseline() -> dict:
    """Measure Groq API latency for baseline comparison."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("\n--- Groq baseline (llama-3.3-70b-versatile) ---")
    latencies, token_counts = [], []

    for i, q in enumerate(QUERIES[:5]):   # 5 queries to avoid rate limits
        t0 = time.time()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": q},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        elapsed = time.time() - t0
        tokens  = resp.usage.completion_tokens
        latencies.append(elapsed)
        token_counts.append(tokens)
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  Q{i+1}: {elapsed*1000:.0f}ms  {tokens} tokens  {tps:.1f} tok/s")
        time.sleep(0.5)  # rate limit buffer

    avg_latency = sum(latencies) / len(latencies)
    total_tokens = sum(token_counts)
    total_time   = sum(latencies)
    return {
        "model":         "llama-3.3-70b-versatile (Groq API)",
        "n_queries":     len(latencies),
        "avg_latency_s": round(avg_latency, 3),
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "total_tokens":  total_tokens,
        "total_time_s":  round(total_time, 2),
        "throughput_tps": round(total_tokens / total_time, 1) if total_time > 0 else 0,
    }


def run_local_dpo_benchmark() -> dict:
    """Benchmark DPO checkpoint on RTX 4090 using HF generate()."""
    if not DPO_DIR.exists():
        print(f"DPO checkpoint not found at {DPO_DIR}")
        return {"error": "DPO checkpoint missing"}

    print(f"\n--- Local DPO benchmark (Qwen2.5-1.5B + LoRA, {DPO_DIR.name}) ---")
    print("Loading model in 4-bit NF4...")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    t_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(DPO_DIR))
    model.eval()

    load_time = time.time() - t_load
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Model loaded in {load_time:.1f}s  VRAM: {vram_gb:.1f} GB")

    latencies, token_counts, ttfts = [], [], []

    for i, q in enumerate(QUERIES):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Measure total generation time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - t0

        n_new = output_ids.shape[1] - inputs.input_ids.shape[1]
        tps   = n_new / elapsed if elapsed > 0 else 0
        latencies.append(elapsed)
        token_counts.append(n_new)

        # TTFT approximation: time per token × 1
        ttft_approx = elapsed / n_new if n_new > 0 else elapsed
        ttfts.append(ttft_approx)

        print(f"  Q{i+1}: {elapsed*1000:.0f}ms  {n_new} tokens  {tps:.1f} tok/s  TTFT~{ttft_approx*1000:.0f}ms")

    avg_latency  = sum(latencies) / len(latencies)
    total_tokens = sum(token_counts)
    total_time   = sum(latencies)
    avg_tps      = total_tokens / total_time if total_time > 0 else 0
    avg_ttft     = sum(ttfts) / len(ttfts)

    return {
        "model":          f"Qwen2.5-1.5B-Instruct + LoRA DPO ({DPO_DIR.name})",
        "quantization":   "4-bit NF4 (bitsandbytes)",
        "hardware":       "RTX 4090 Laptop GPU (16GB VRAM)",
        "note":           "HuggingFace generate() — sequential, not batched (vLLM Linux-only)",
        "n_queries":      len(latencies),
        "model_load_s":   round(load_time, 1),
        "vram_gb":        round(vram_gb, 2),
        "avg_latency_s":  round(avg_latency, 3),
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "avg_ttft_ms":    round(avg_ttft * 1000, 1),
        "total_tokens":   total_tokens,
        "total_time_s":   round(total_time, 2),
        "throughput_tps": round(avg_tps, 1),
    }


def main():
    print("=" * 65)
    print("vLLM Throughput Benchmark — India Steel Trade Intelligence Platform")
    print("=" * 65)
    print()
    print("NOTE: vLLM is Linux-only (no native Windows support).")
    print("Benchmarking DPO checkpoint via HuggingFace generate() on RTX 4090.")
    print("vLLM PagedAttention would improve throughput 2-4x on Linux/WSL2.")
    print()

    results = {}

    # Local DPO model
    try:
        results["local_dpo"] = run_local_dpo_benchmark()
    except Exception as e:
        print(f"[ERROR] Local DPO benchmark failed: {e}")
        results["local_dpo"] = {"error": str(e)}

    # Groq baseline
    try:
        results["groq_baseline"] = run_groq_baseline()
    except Exception as e:
        print(f"[ERROR] Groq baseline failed: {e}")
        results["groq_baseline"] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 65)
    print("THROUGHPUT COMPARISON SUMMARY")
    print("=" * 65)
    local = results.get("local_dpo", {})
    groq  = results.get("groq_baseline", {})

    print(f"\nLocal DPO (RTX 4090, HF generate(), 4-bit NF4):")
    if "error" not in local:
        print(f"  Avg latency : {local.get('avg_latency_ms')} ms")
        print(f"  TTFT        : ~{local.get('avg_ttft_ms')} ms (per-token approx)")
        print(f"  Throughput  : {local.get('throughput_tps')} tokens/sec")
        print(f"  VRAM usage  : {local.get('vram_gb')} GB")
    else:
        print(f"  Error: {local['error']}")

    print(f"\nGroq API (llama-3.3-70b-versatile, cloud):")
    if "error" not in groq:
        print(f"  Avg latency : {groq.get('avg_latency_ms')} ms")
        print(f"  Throughput  : {groq.get('throughput_tps')} tokens/sec")
    else:
        print(f"  Error: {groq['error']}")

    print(f"\nvLLM note: On Linux/WSL2, PagedAttention + continuous batching")
    print(f"typically achieves 2-4x higher throughput vs HF generate().")
    print(f"Production decision: Groq remains the serving path (no latency penalty,")
    print(f"no GPU memory, handles concurrent requests via cloud infrastructure).")

    out_path = Path(__file__).parent / "throughput_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
