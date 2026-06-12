"""
train_dpo_local.py — DPO fine-tuning on local GPU (RTX 4090 Laptop, 16GB VRAM).

Model  : Qwen/Qwen2.5-1.5B-Instruct (4-bit NF4)
Dataset: dpo/preference_pairs.json (70 pairs)
Output : dpo/dpo_checkpoint/
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
PAIRS_PATH = Path(__file__).parent / "preference_pairs.json"
CKPT_DIR   = Path(__file__).parent / "dpo_checkpoint"
EVAL_OUT   = Path(__file__).parent / "dpo_eval_results.json"

MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a steel trade policy analyst specialising in India. "
    "Answer questions about Indian steel trade policy using only the context provided. "
    "Always cite the specific regulation, investigation, or document that supports your answer. "
    "If you cannot find the answer in the provided context, say so explicitly."
)

# ── GPU check ─────────────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("ERROR: No CUDA GPU found. Run with anaconda python.")
    sys.exit(1)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print(f"\nLoading pairs from {PAIRS_PATH}")
with open(PAIRS_PATH) as f:
    pairs = json.load(f)
print(f"Loaded {len(pairs)} preference pairs")

def format_pair(p):
    return {
        "prompt":   [{"role": "system",    "content": SYSTEM_PROMPT},
                     {"role": "user",      "content": p["question"]}],
        "chosen":   [{"role": "assistant", "content": p["chosen"]}],
        "rejected": [{"role": "assistant", "content": p["rejected"]}],
    }

formatted = [format_pair(p) for p in pairs]
split     = int(len(formatted) * 0.8)
train_ds  = Dataset.from_list(formatted[:split])
test_ds   = Dataset.from_list(formatted[split:])
print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}")

# ── 2. Load model in 4-bit ────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} in 4-bit NF4 …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 3. LoRA ───────────────────────────────────────────────────────────────────
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 4. DPO training ───────────────────────────────────────────────────────────
print("\nStarting DPO training …")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

training_args = DPOConfig(
    output_dir=str(CKPT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    beta=0.1,
    max_length=1024,
    logging_steps=5,
    save_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    bf16=True,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
)

trainer.train()
print("Training complete.")

# ── 5. Save checkpoint ────────────────────────────────────────────────────────
trainer.save_model(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))
print(f"Checkpoint saved to {CKPT_DIR}")

# ── 6. Faithfulness eval vs Groq baseline ─────────────────────────────────────
print("\nRunning NLI faithfulness eval on 8 ground-truth questions …")
from sentence_transformers import CrossEncoder
import re

EVAL_QUESTIONS = [
    "Which countries are subject to anti-dumping investigation on electrogalvanized steel imports into India?",
    "What products are covered under the anti-dumping investigation on seamless tubes from China?",
    "How many countries are named in the anti-dumping investigation on flat rolled products of stainless steel?",
    "What type of products are covered by the safeguard investigation on steel flat products?",
    "What does PCN stand for and why is it used in anti-dumping investigations?",
    "What is the difference between an anti-dumping measure and a safeguard measure?",
    "What is the IS 2062 standard and which steel products does it apply to?",
    "What triggers the initiation of a safeguard investigation in India?",
]

CONTEXT = (
    "Context: India uses DGTR for trade remedy investigations. "
    "Anti-dumping duties require material injury finding. "
    "Safeguard measures apply to all countries. "
    "IS 2062 covers structural steel products. "
    "PCN (Product Control Number) standardises product comparison in AD investigations."
)

dpo_answers = []
model.eval()
for q in EVAL_QUESTIONS:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"{CONTEXT}\n\nQuestion: {q}"},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    dpo_answers.append({"question": q, "answer": answer})
    print(f"Q: {q[:70]}")
    print(f"A: {answer[:120]}\n")

with open(EVAL_OUT, "w") as f:
    json.dump(dpo_answers, f, indent=2)

# NLI faithfulness
nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")

def nli_faith(context, answer):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s) > 10]
    if not sentences:
        return 1.0
    scores = nli.predict([(context, s) for s in sentences], apply_softmax=True)
    contradicted = sum(1 for s in scores if s.argmax() == 0)
    return round((len(sentences) - contradicted) / len(sentences), 3)

faithfulness_scores = [nli_faith(CONTEXT, r["answer"]) for r in dpo_answers]
avg = sum(faithfulness_scores) / len(faithfulness_scores)
groq_baseline = 1.00

print("\n=== GATE TEST ===")
for i, (r, sc) in enumerate(zip(dpo_answers, faithfulness_scores)):
    print(f"  [{i+1}] NLI faith={sc:.3f}  {r['question'][:60]}")
print(f"\nDPO avg NLI faithfulness : {avg:.3f}")
print(f"Groq v3 baseline         : {groq_baseline:.3f}")
print(f"Delta                    : {avg - groq_baseline:+.3f}")
if avg >= groq_baseline - 0.05:
    print("DECISION: DPO model acceptable for production")
else:
    print("DECISION: Keep Groq in production — document DPO result in benchmark report")

print(f"\nEval results saved to {EVAL_OUT}")
