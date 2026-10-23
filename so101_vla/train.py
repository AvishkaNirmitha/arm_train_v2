"""Fine-tune Qwen3.5-2B (LoRA, Unsloth) to output SO-101 joint keyframes.
Same recipe as ../vlm_training/train_pick_place.py (the Panda model), with SO-101 data and folders.

    python train.py --max-steps 30                  # smoke test
    python train.py                                 # full run (1 epoch)
    python train.py --resume checkpoints/<run>      # continue an interrupted run

Every run gets its own folder:
    checkpoints/<name>_<dd-mm-yyyy-hh-mm-ss>/
        config.json    run settings
        trainer/       intermediate checkpoints (for --resume, and usable with run_policy.py)
        lora/          final LoRA adapters + tokenizer

There is no offline eval at the end: for the Panda it was misleading. Measure with run_policy.py,
which lets the model drive the arm in closed loop.
"""
import os
import sys
# Unsloth compiles each decoder layer as one region. For Qwen3.5's linear-attention layers
# (pure-PyTorch fallback: causal_conv1d / fla not installed) that compile runs on the CPU for
# ages with the GPU idle. Off: ~3 s/step on an RTX 4060 laptop instead of never starting.
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")

import unsloth  # noqa: F401  (must be imported before transformers / trl)

import glob
import json
import argparse
from datetime import datetime

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))  # datasets/ and checkpoints/ live here
RUNNER = "run_policy.py"                            # script named in the final hint
sys.path.append(os.path.join(os.path.dirname(HERE), "vlm_training"))
from pick_place_dataset import PickPlaceConversations  # noqa: E402  (generic jsonl + image loader)


def latest_dataset():
    dirs = [d for d in glob.glob(os.path.join(HERE, "datasets", "*"))
            if os.path.exists(os.path.join(d, "train.jsonl"))]
    if not dirs:
        raise SystemExit("no dataset found in so101_vla/datasets/ - run generate_dataset.py first")
    return max(dirs, key=os.path.getmtime)


def gpu_mem_gb():
    return round(torch.cuda.max_memory_reserved() / 2**30, 2)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--dataset", default=None, help="dataset dir (default: newest in so101_vla/datasets/)")
    ap.add_argument("--name", default="qwen35_2b_so101", help="run folder prefix")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue a run from its last checkpoint")
    ap.add_argument("--cams", default=None, help="comma list, e.g. 'top' (default: all saved cameras)")
    ap.add_argument("--load-in-4bit", action="store_true", help="use if you run out of GPU memory")

    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when > 0")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.resume:
        run_dir = os.path.abspath(args.resume)
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
        for k, v in saved["args"].items():  # keep the original run's settings
            if k != "resume":
                setattr(args, k, v)
    else:
        args.dataset = os.path.abspath(args.dataset or latest_dataset())
        run_dir = os.path.join(HERE, "checkpoints", f"{args.name}_{datetime.now():%d-%m-%Y-%H-%M-%S}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"args": vars(args), "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    cams = args.cams.split(",") if args.cams else None
    print(f"[train] run dir : {run_dir}")
    print(f"[train] dataset : {args.dataset}")
    print(f"[train] model   : {args.model}  (4bit={args.load_in_4bit})")

    # ── Model + LoRA ───────────────────────────────────────────────────────
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,     # reading exact cube pose from pixels is the hard part
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )

    # ── Data (prompts are used exactly as stored: image-only, no current joints) ──
    train_dataset = PickPlaceConversations(args.dataset, split="train", cams=cams).to_hf_dataset()
    print(f"[train] {len(train_dataset)} train samples")

    # ── Train ──────────────────────────────────────────────────────────────
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=os.path.join(run_dir, "trainer"),
            report_to="none",

            # Required for vision fine-tuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    print(f"[train] GPU {gpu.name}, {gpu.total_memory / 2**30:.1f} GB total, {gpu_mem_gb()} GB reserved before training")
    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)
    print(f"[train] done in {stats.metrics['train_runtime'] / 60:.1f} min, "
          f"final loss {stats.metrics.get('train_loss', float('nan')):.4f}, peak GPU memory {gpu_mem_gb()} GB")

    lora_dir = os.path.join(run_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"[train] LoRA adapters saved to {lora_dir}")
    print(f"[train] try it:  python {RUNNER} --lora {os.path.relpath(lora_dir, HERE)} --episodes 10")


if __name__ == "__main__":
    main()
