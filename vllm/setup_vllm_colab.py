"""
setup_vllm_colab.py — vLLM serving setup for the Steel RAG Platform.

Run this script in Google Colab (T4/A100) to serve any compatible model
as an OpenAI-compatible endpoint. The platform's router.py can then use
the vLLM endpoint as a local alternative to Groq for offline/cost-saving mode.

Steps:
  1. Run cell 1: install vllm
  2. Run cell 2: start vLLM server in background (Qwen2.5-1.5B or larger model)
  3. Run cell 3: expose via ngrok public URL
  4. Run cell 4: test with the platform's router

Note: Colab T4 can serve models up to ~7B (4-bit). A100 supports 70B.
Groq remains the production path — this is for offline/demo use.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Install vLLM
# ═══════════════════════════════════════════════════════════════════════════════
CELL_1 = """
!pip install vllm -q
!pip install pyngrok -q
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Start vLLM server
# ═══════════════════════════════════════════════════════════════════════════════
CELL_2 = """
import subprocess, time, os

# Choose model based on available VRAM:
#   T4  (15GB): Qwen2.5-7B-Instruct-GPTQ-Int4 (7B, 4-bit ~5GB)
#   A100(40GB): Qwen2.5-14B-Instruct or meta-llama/Llama-3.1-8B-Instruct

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model",          MODEL_ID,
    "--host",           "0.0.0.0",
    "--port",           "8080",
    "--max-model-len",  "4096",
    "--dtype",          "half",
    "--gpu-memory-utilization", "0.90",
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(f"vLLM server PID: {proc.pid}")

# Wait for server ready
import time
for i in range(60):
    time.sleep(5)
    if proc.poll() is not None:
        out, _ = proc.communicate()
        print("Server crashed:", out.decode()[-2000:])
        break
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8080/health", timeout=2)
        print("vLLM server is READY at http://localhost:8080")
        break
    except Exception:
        print(f"  Waiting {(i+1)*5}s …")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Expose via ngrok
# ═══════════════════════════════════════════════════════════════════════════════
CELL_3 = """
from pyngrok import ngrok

# Paste your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
ngrok.set_auth_token(NGROK_TOKEN)

public_url = ngrok.connect(8080)
print(f"vLLM public endpoint: {public_url}")
print()
print("Set this in your .env or pass to router:")
print(f"  VLLM_BASE_URL={public_url}/v1")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Test with platform router
# ═══════════════════════════════════════════════════════════════════════════════
CELL_4 = """
import openai

VLLM_URL = "http://localhost:8080/v1"   # or paste ngrok URL
MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

client = openai.OpenAI(base_url=VLLM_URL, api_key="none")

resp = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system",  "content": "You are a steel trade policy analyst."},
        {"role": "user",    "content": "What is India's anti-dumping duty on HR coil from China?"},
    ],
    temperature=0.1,
    max_tokens=200,
)
print("Response:", resp.choices[0].message.content)
print("Tokens used:", resp.usage.total_tokens)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: How to use vLLM endpoint in router.py
# ═══════════════════════════════════════════════════════════════════════════════
CELL_5 = """
# To use the vLLM endpoint instead of Groq in the platform's router.py:
#
# 1. Set environment variable:
#    VLLM_BASE_URL=http://localhost:8080/v1   (or ngrok URL)
#
# 2. In steel_rag/rag.py or router.py, the Groq client can be swapped:
#
#    import os
#    from openai import OpenAI
#
#    VLLM_URL = os.getenv("VLLM_BASE_URL")
#    if VLLM_URL:
#        client = OpenAI(base_url=VLLM_URL, api_key="none")
#        model  = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
#    else:
#        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#        model  = "llama-3.3-70b-versatile"
#
# 3. Both clients use the same OpenAI-compatible .chat.completions.create() interface.
"""

# ── Print all cells for easy copy-paste into Colab ──────────────────────────
if __name__ == "__main__":
    cells = [
        ("Install vLLM", CELL_1),
        ("Start vLLM server", CELL_2),
        ("Expose via ngrok", CELL_3),
        ("Test with OpenAI client", CELL_4),
        ("Integrate with router.py", CELL_5),
    ]
    print("=" * 70)
    print("vLLM Colab Setup — India Steel Trade Intelligence Platform")
    print("Copy each block into a separate Colab cell")
    print("=" * 70)
    for i, (title, code) in enumerate(cells, 1):
        print(f"\n{'─'*60}")
        print(f"CELL {i}: {title}")
        print('─'*60)
        print(code.strip())
