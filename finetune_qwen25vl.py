"""
LoRA Fine-Tuning — Qwen2.5-VL-7B-Instruct
==========================================
Fine-tunes Qwen2.5-VL-7B on the auto-labelled droplet impact dataset.

Setup
-----
  • Model  : Qwen/Qwen2.5-VL-7B-Instruct
  • Method : LoRA (rank 16, targeting LM attention + FFN layers)
  • Epochs : 3
  • Batch  : 1 per GPU × 8 gradient-accumulation steps → effective 8
  • Precision: bf16 (A100 native)
  • Checkpoint: finetune_data/qwen25vl_lora/

Usage
-----
    /opt/anaconda3/2024.02-1/conda_envs/ml_dl_gpu_base/bin/python \
        finetune_qwen25vl.py
"""

import json
import base64
import io
import os
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR    = Path("/home/ubuntu/materials/finetune_data")
OUT_DIR     = DATA_DIR / "qwen25vl_lora"
TRAIN_JSONL = DATA_DIR / "finetune_train.jsonl"
VAL_JSONL   = DATA_DIR / "finetune_val.jsonl"

EPOCHS          = 3
BATCH_SIZE      = 1
GRAD_ACCUM      = 8     # effective batch = 8
LR              = 2e-4
MAX_SEQ_LEN     = 2048
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
WARMUP_RATIO    = 0.03
LOG_STEPS       = 5
SAVE_STEPS      = 50
EVAL_STEPS      = 50
SEED            = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
def b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class DropletDataset(Dataset):
    """
    Each JSONL entry has 'messages': [system, user, assistant].
    The user message has an image_url (base64 JPEG) and a text prompt.
    We convert to the format expected by Qwen2.5-VL's processor.
    """
    def __init__(self, jsonl_path: Path, processor):
        self.records   = load_jsonl(jsonl_path)
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        msgs = self.records[idx]["messages"]   # [system, user, assistant]

        # ── Extract image from user message ──────────────────────────────────
        user_content = msgs[1]["content"]      # list of {type, image_url/text}
        image = None
        text_parts = []
        for part in user_content:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                b64 = url.split("base64,", 1)[1]
                image = b64_to_pil(b64)
            elif part["type"] == "text":
                text_parts.append(part["text"])
        user_text = "\n".join(text_parts)

        system_text   = msgs[0]["content"]
        assistant_text = msgs[2]["content"]

        # ── Build Qwen2.5-VL chat messages ───────────────────────────────────
        # The processor expects PIL images passed in-line via a special token
        qwen_messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": user_text},
                ],
            },
            {"role": "assistant", "content": assistant_text},
        ]

        return qwen_messages, image


def collate_fn(batch, processor):
    """Tokenise a batch; mask the prompt tokens in labels."""
    all_messages = [item[0] for item in batch]
    all_images   = [item[1] for item in batch]

    # Build text for each example using the chat template
    # apply_chat_template returns the full string including assistant turn
    texts = [
        processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        for msgs in all_messages
    ]

    # Tokenise with images
    inputs = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    # Build labels: mask everything before the assistant turn
    # Find the separator token(s) that precede the assistant reply
    input_ids = inputs["input_ids"]
    labels    = input_ids.clone()

    # Mask padding
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask all tokens up to and including the assistant start token.
    # The assistant turn in Qwen starts with "<|im_start|>assistant\n".
    # We iterate each example in the batch.
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

    for i in range(len(texts)):
        ids = input_ids[i].tolist()
        # Find last occurrence of [im_start, assistant] pair
        mask_until = 0
        for j in range(len(ids) - 1):
            if ids[j] == im_start_id and ids[j + 1] == assistant_id:
                mask_until = j + 2   # mask up to (and including) "assistant\n"
        labels[i, :mask_until] = -100

    inputs["labels"] = labels
    return inputs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor and model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=1280*28*28,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    # Target attention + FFN in the language model layers.
    # The vision encoder weights are frozen (no LoRA) to preserve its
    # pretrained spatial understanding; only the LM needs to learn the
    # JSON output format and pixel-value estimation.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        bias="none",
    )

    # ── Training config ───────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_seq_length=MAX_SEQ_LEN,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = DropletDataset(TRAIN_JSONL, processor)
    val_ds   = DropletDataset(VAL_JSONL,   processor)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    import functools
    _collate = functools.partial(collate_fn, processor=processor)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=_collate,
    )

    print("Starting training …")
    trainer.train()

    # Save final adapter
    final_path = OUT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nFinal adapter saved → {final_path}")

    # Training summary
    log_history = trainer.state.log_history
    train_losses = [(e["step"], e["loss"])
                    for e in log_history if "loss" in e]
    eval_losses  = [(e["step"], e["eval_loss"])
                    for e in log_history if "eval_loss" in e]

    print("\n── Training loss (last 10 entries) ──")
    for step, loss in train_losses[-10:]:
        print(f"  step {step:4d}: {loss:.4f}")
    print("\n── Eval loss per checkpoint ──")
    for step, loss in eval_losses:
        print(f"  step {step:4d}: {loss:.4f}")


if __name__ == "__main__":
    main()
