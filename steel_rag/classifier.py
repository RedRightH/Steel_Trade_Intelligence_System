"""
classifier.py - 5-class steel trade news headline classifier.

Labels:
  ANTI_DUMPING        - AD duties, dumping margins, DGTR investigations
  SAFEGUARD           - safeguard measures, surge in imports, serious injury
  RAW_MATERIAL        - coking coal, iron ore, scrap, input costs
  POLICY_OPPORTUNITY  - FTAs, PLI scheme, export incentives, new markets
  CBAM_COMPLIANCE     - EU CBAM, carbon border, green steel, emissions

Two modes:
  1. Zero-shot  - facebook/bart-large-mnli  (no training needed, baseline)
  2. Fine-tuned - distilbert-base-uncased   (trained on synthetic + real data)

Usage:
  python classifier.py --mode zeroshot    # run zero-shot on gate test cases
  python classifier.py --mode train       # fine-tune and evaluate
  python classifier.py --mode test        # run gate test on fine-tuned model
  python classifier.py                    # run full pipeline (zero-shot -> train -> gate test)
"""

import sys
import json
import time
import argparse
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABELS = ["ANTI_DUMPING", "SAFEGUARD", "RAW_MATERIAL", "POLICY_OPPORTUNITY", "CBAM_COMPLIANCE"]
MODEL_DIR = Path(__file__).parent / "classifier_model"
DATA_DIR  = Path(__file__).parent / "eval"

# ── Training data ──────────────────────────────────────────────────────────────
# 12 headlines per class (synthetic + representative real examples)
TRAINING_DATA = [
    # ANTI_DUMPING
    ("India imposes anti-dumping duty on seamless steel pipes from China", "ANTI_DUMPING"),
    ("DGTR recommends dumping margin of 18.5% on Chinese HR coil imports", "ANTI_DUMPING"),
    ("Electrogalvanized steel from Korea faces anti-dumping investigation in India", "ANTI_DUMPING"),
    ("India extends anti-dumping duty on stainless steel from China for 5 years", "ANTI_DUMPING"),
    ("DGTR initiates sunset review of AD duty on cold-rolled steel from Japan", "ANTI_DUMPING"),
    ("India probes dumping of alloy steel pipes from Vietnam and Thailand", "ANTI_DUMPING"),
    ("Anti-dumping duty on hot-rolled coils from China upheld by CESTAT", "ANTI_DUMPING"),
    ("India files WTO case against US Section 232 steel tariffs", "ANTI_DUMPING"),
    ("Dumping of Chinese galvanized steel causes injury to domestic producers: DGTR", "ANTI_DUMPING"),
    ("New anti-dumping investigation on colour-coated steel sheets from China and Korea", "ANTI_DUMPING"),
    ("India seeks consultations at WTO on EU steel anti-dumping measures", "ANTI_DUMPING"),
    ("DGTR final findings confirm dumping of stainless steel from Indonesia", "ANTI_DUMPING"),

    # SAFEGUARD
    ("India imposes 25% safeguard duty on steel flat products for 200 days", "SAFEGUARD"),
    ("DGTR preliminary findings show serious injury from surge in steel imports", "SAFEGUARD"),
    ("Safeguard investigation initiated on non-alloy steel flat products entering India", "SAFEGUARD"),
    ("India extends safeguard measures on HR steel products citing continued injury", "SAFEGUARD"),
    ("Steel imports surge 40% in Q1, industry seeks emergency safeguard relief", "SAFEGUARD"),
    ("DGTR recommends provisional safeguard duty on cold-rolled steel sheets", "SAFEGUARD"),
    ("Domestic steel industry files safeguard petition with DGTR over import surge", "SAFEGUARD"),
    ("India's safeguard duty on steel to be reviewed after WTO dispute", "SAFEGUARD"),
    ("Safeguard measure on galvanized steel extended by two years", "SAFEGUARD"),
    ("Surge in steel plate imports from China triggers safeguard probe in India", "SAFEGUARD"),
    ("WTO Appellate Body rules on India steel safeguard measure compliance", "SAFEGUARD"),
    ("India considers blanket safeguard on all steel flat products for 3 years", "SAFEGUARD"),

    # RAW_MATERIAL
    ("Australian coking coal exports to India fall amid port congestion", "RAW_MATERIAL"),
    ("Iron ore prices hit 6-month high as China demand surges", "RAW_MATERIAL"),
    ("India increases coking coal imports from USA as Australia supply tightens", "RAW_MATERIAL"),
    ("Global scrap steel prices rise sharply on Turkish buying spree", "RAW_MATERIAL"),
    ("BHP warns of coking coal supply disruption due to Queensland floods", "RAW_MATERIAL"),
    ("India's iron ore export ban lifted, impacting domestic steel input costs", "RAW_MATERIAL"),
    ("Coking coal shortage threatens Indian steel production capacity", "RAW_MATERIAL"),
    ("Scrap steel imports to India up 30% as domestic availability tightens", "RAW_MATERIAL"),
    ("Russia-Ukraine war disrupts iron ore and scrap supply to Indian steelmakers", "RAW_MATERIAL"),
    ("Steel input costs rise as coking coal hits $350 per tonne", "RAW_MATERIAL"),
    ("India's dependence on Australian coking coal: risks and alternatives", "RAW_MATERIAL"),
    ("Port strike in Australia delays coking coal cargo to Indian steel mills", "RAW_MATERIAL"),

    # POLICY_OPPORTUNITY
    ("India-UAE CEPA opens new market for Indian flat steel exports", "POLICY_OPPORTUNITY"),
    ("PLI scheme for specialty steel attracts Rs 6,000 crore investment", "POLICY_OPPORTUNITY"),
    ("India signs trade agreement with Australia opening steel export opportunities", "POLICY_OPPORTUNITY"),
    ("DGFT simplifies steel export procedures under new Foreign Trade Policy 2023", "POLICY_OPPORTUNITY"),
    ("India targets EU market with green steel exports under EFTA deal", "POLICY_OPPORTUNITY"),
    ("PLI for specialty steel: 27 companies selected, production targets set", "POLICY_OPPORTUNITY"),
    ("India eyes 150 million tonne steel capacity by 2030 under National Steel Policy", "POLICY_OPPORTUNITY"),
    ("RCEP exit: India's steel sector gains competitive advantage in Southeast Asia", "POLICY_OPPORTUNITY"),
    ("India-GCC free trade agreement to boost steel exports to Gulf markets", "POLICY_OPPORTUNITY"),
    ("Ministry of Steel launches export promotion council for specialty grades", "POLICY_OPPORTUNITY"),
    ("New advance authorisation scheme to cut steel export costs by 12%", "POLICY_OPPORTUNITY"),
    ("India's HR coil exports to EU surge after bilateral trade talks", "POLICY_OPPORTUNITY"),

    # CBAM_COMPLIANCE
    ("EU carbon border adjustment mechanism enters transition phase for steel exports", "CBAM_COMPLIANCE"),
    ("Indian steel exporters must report embedded carbon under EU CBAM rules", "CBAM_COMPLIANCE"),
    ("CBAM will add 15-20 EUR/tonne cost to Indian steel exports to Europe", "CBAM_COMPLIANCE"),
    ("Green steel investments rise as Indian mills prepare for CBAM compliance", "CBAM_COMPLIANCE"),
    ("EU CBAM reporting deadline: Indian steel firms scramble for carbon data", "CBAM_COMPLIANCE"),
    ("Carbon intensity of Indian steel sector must fall to meet EU CBAM thresholds", "CBAM_COMPLIANCE"),
    ("SAIL and Tata Steel invest in hydrogen steelmaking for CBAM-free exports", "CBAM_COMPLIANCE"),
    ("India-EU trade talks include CBAM carve-out for developing country steel", "CBAM_COMPLIANCE"),
    ("CBAM implementation to reshape India's steel export basket toward EU markets", "CBAM_COMPLIANCE"),
    ("Indian steelmakers adopt scrap-based EAF route to reduce CBAM liability", "CBAM_COMPLIANCE"),
    ("Bureau of Energy Efficiency certifies Indian steel plants for CBAM carbon reporting", "CBAM_COMPLIANCE"),
    ("CBAM financial impact on Indian steel sector estimated at $500 million annually", "CBAM_COMPLIANCE"),
]

# ── Gate test cases (one per class, mandatory) ─────────────────────────────────
GATE_TEST_CASES = [
    ("DGTR recommends anti-dumping duty on Chinese HR coil imports", "ANTI_DUMPING"),
    ("India imposes provisional safeguard duty on steel flat products citing surge", "SAFEGUARD"),
    ("Coking coal shortage in Australia disrupts Indian steel mill supply chain", "RAW_MATERIAL"),
    ("India-UAE CEPA steel concessions take effect boosting flat steel exports", "POLICY_OPPORTUNITY"),
    ("Indian steel exporters register for EU CBAM carbon reporting obligation", "CBAM_COMPLIANCE"),
]


# ── Zero-shot classifier ───────────────────────────────────────────────────────

def zero_shot_classify(headline: str) -> dict:
    """Classify using facebook/bart-large-mnli zero-shot pipeline."""
    from transformers import pipeline

    if not hasattr(zero_shot_classify, "_pipe"):
        print("Loading zero-shot model (facebook/bart-large-mnli, ~1.6GB)...")
        zero_shot_classify._pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        print("Zero-shot model loaded.")

    candidate_labels = [
        "anti-dumping duty investigation",
        "safeguard measure import surge",
        "raw material supply coking coal iron ore scrap",
        "trade policy opportunity export FTA PLI scheme",
        "carbon border adjustment CBAM green steel compliance",
    ]

    result = zero_shot_classify._pipe(headline, candidate_labels, multi_label=False)
    best_idx = result["scores"].index(max(result["scores"]))
    predicted_label = LABELS[best_idx]
    confidence = result["scores"][best_idx]

    return {"label": predicted_label, "confidence": round(confidence, 3)}


# ── Fine-tuned classifier ──────────────────────────────────────────────────────

def train_classifier():
    """Fine-tune distilbert-base-uncased on the training data."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    import random

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for i, l in enumerate(LABELS)}

    random.shuffle(TRAINING_DATA)
    split = int(len(TRAINING_DATA) * 0.8)
    train_data = TRAINING_DATA[:split]
    val_data   = TRAINING_DATA[split:]

    print(f"Training: {len(train_data)} | Validation: {len(val_data)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class HeadlineDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            text, label = self.data[idx]
            enc = tokenizer(text, truncation=True, padding="max_length",
                            max_length=64, return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(label2id[label]),
            }

    train_loader = DataLoader(HeadlineDataset(train_data), batch_size=4, shuffle=True)
    val_loader   = DataLoader(HeadlineDataset(val_data),   batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 20
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nFine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()

        val_acc = correct / len(val_data)
        print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.3f} | val_acc={val_acc:.2%}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModel saved to {MODEL_DIR}/")

    return model, tokenizer, val_acc


def load_finetuned():
    """Load the fine-tuned model."""
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Fine-tuned model not found. Run: python classifier.py --mode train")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return model, tokenizer


def finetuned_classify(headline: str, model=None, tokenizer=None) -> dict:
    """Classify using the fine-tuned DistilBERT model."""
    import torch
    if model is None or tokenizer is None:
        model, tokenizer = load_finetuned()

    inputs = tokenizer(headline, return_tensors="pt", truncation=True,
                       padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label":      LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 3),
        "all_scores": {l: round(float(probs[i]), 3) for i, l in enumerate(LABELS)},
    }


# ── Gate test ──────────────────────────────────────────────────────────────────

def run_gate_test(mode: str = "finetuned"):
    """Run 5 mandatory gate test cases. All must pass."""
    print("\n" + "=" * 60)
    print(f"GATE TEST ({mode}) - 5 mandatory cases")
    print("=" * 60)

    if mode == "finetuned":
        model, tokenizer = load_finetuned()
        classify_fn = lambda h: finetuned_classify(h, model, tokenizer)
    else:
        classify_fn = zero_shot_classify

    passed = 0
    for headline, expected in GATE_TEST_CASES:
        result = classify_fn(headline)
        ok = result["label"] == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] Expected={expected}")
        print(f"         Got={result['label']} (conf={result['confidence']:.2f})")
        print(f"         '{headline[:65]}'")
        print()

    print(f"Gate result: {passed}/5 passed")
    gate_ok = passed == 5
    print(f"Gate status: {'PASS' if gate_ok else 'FAIL - do not connect to corpus pipeline'}")
    return gate_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeroshot", "train", "test", "all"],
                        default="all")
    args = parser.parse_args()

    if args.mode in ("zeroshot", "all"):
        print("=" * 60)
        print("ZERO-SHOT BASELINE (facebook/bart-large-mnli)")
        print("=" * 60)
        correct = 0
        for headline, expected in GATE_TEST_CASES:
            result = zero_shot_classify(headline)
            ok = result["label"] == expected
            if ok:
                correct += 1
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {result['label']} (conf={result['confidence']:.2f}) | expected={expected}")
        print(f"\nZero-shot accuracy on gate cases: {correct}/5\n")

    if args.mode in ("train", "all"):
        print("=" * 60)
        print("FINE-TUNING distilbert-base-uncased")
        print("=" * 60)
        _, _, val_acc = train_classifier()
        print(f"\nFinal validation accuracy: {val_acc:.2%}")
        target = 0.75
        print(f"Target >= {target:.0%}: {'PASS' if val_acc >= target else 'FAIL'}")

    if args.mode in ("test", "all"):
        run_gate_test("finetuned")


if __name__ == "__main__":
    main()
