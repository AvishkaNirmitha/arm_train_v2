"""Time and profile single training micro-steps, outside the notebook.

Same model / LoRA / collator settings as Qwen3_5_(2B)_PickPlace_SFT.ipynb. Needs the GPU free
(shut down the notebook kernel first). Never downloads: the model must already be cached.

    python bench_train_step.py                              # Qwen3.5-2B
    python bench_train_step.py --model unsloth/Qwen3-VL-2B-Instruct
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see train_pick_place.py

import unsloth  # noqa: F401  (must be imported before transformers/trl)
import time
import argparse

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from pick_place_dataset import PickPlaceConversations

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../datasets/pick_place_v1_21-10-2026-11-03-15")


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)

    # Which implementation do the linear-attention layers use?
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith("GatedDeltaNet"):
            fn = getattr(mod, "chunk_gated_delta_rule", None)
            print(f"[bench] {type(mod).__name__}: chunk_gated_delta_rule = "
                  f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', fn)}")
            break

    ds = PickPlaceConversations(DATASET_DIR, "train").to_hf_dataset()
    collator = UnslothVisionDataCollator(model, tokenizer)

    def micro_step(i):
        batch, t_collate = timed(lambda: collator([ds[i * args.batch + k] for k in range(args.batch)]))
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        out, t_fwd = timed(lambda: model(**batch))
        _, t_bwd = timed(lambda: out.loss.backward())
        model.zero_grad(set_to_none=True)
        seq = batch["input_ids"].shape[1]
        print(f"[bench] step {i}: seq {seq}  collate {t_collate * 1000:.0f} ms  forward {t_fwd:.2f} s  "
              f"backward {t_bwd:.2f} s  loss {out.loss.item():.3f}", flush=True)
        return t_fwd + t_bwd

    times = [micro_step(i) for i in range(args.steps)]
    steady = sum(times[1:]) / max(len(times) - 1, 1)
    print(f"[bench] first micro-step {times[0]:.1f} s (includes warm-up), steady {steady:.2f} s per micro-step")
    print(f"[bench] => ~{steady * 4:.1f} s per optimizer step (grad accum 4), "
          f"~{steady * 4 * 2827 / 3600:.1f} h for one epoch")
    print(f"[bench] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    if not args.no_profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            micro_step(args.steps)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    main()
