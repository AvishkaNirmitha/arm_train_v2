"""Load the pick-and-place joint-keyframe dataset (from generate_dataset.py) for Unsloth vision SFT.

In the notebook, replace the LaTeX_OCR dataset cells with:

    from pick_place_dataset import PickPlaceConversations
    converted_dataset = PickPlaceConversations("../datasets/pick_place_v1_21-10-2026-11-03-15", split="train")
    val_dataset       = PickPlaceConversations("../datasets/pick_place_v1_21-10-2026-11-03-15", split="val")

Images are opened lazily per sample, so the whole dataset never sits in RAM.

Prompt styles (applied on load, the dataset files are not changed):
    full        prompt as generated: task, phase, current joints, current gripper
    image_only  current joints / gripper removed, so the model has to find the cube and pad in the
                images. With "full", the model learned to copy the current joints and ignore the images.
"""
import os
import json

from PIL import Image

PROMPT_STYLES = ("full", "image_only")
STATE_LINE_PREFIXES = ("Current joints", "Current gripper")


def apply_prompt_style(prompt, style):
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}, expected one of {PROMPT_STYLES}")
    if style == "full":
        return prompt
    return "\n".join(line for line in prompt.split("\n") if not line.startswith(STATE_LINE_PREFIXES))


def run_prompt_style(lora_dir):
    """Prompt style a trained run used, read from the run's config.json (a lora/ or
    trainer/checkpoint-N dir sits 1-2 levels below it). Runs from before the option: "full"."""
    d = os.path.abspath(lora_dir)
    for _ in range(3):
        cfg = os.path.join(d, "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                saved = json.load(f)
            if "args" in saved:
                return saved["args"].get("prompt_style", "full")
        d = os.path.dirname(d)
    return "full"


class PickPlaceConversations:
    def __init__(self, root, split="train", cams=None, prompt_style="full"):
        self.root = root
        with open(os.path.join(root, f"{split}.jsonl")) as f:
            self.rows = [json.loads(line) for line in f]
        self.cams = cams  # e.g. ["top"] to train on a subset of the saved cameras
        self.prompt_style = prompt_style
        apply_prompt_style("", prompt_style)  # validate early

    def prompt(self, row):
        return apply_prompt_style(row["prompt"], self.prompt_style)

    def __len__(self):
        return len(self.rows)

    def images(self, row):
        paths = row["images"]
        if self.cams is not None:
            paths = [p for p in paths if any(p.endswith(f"_{c}.jpg") for c in self.cams)]
        return [Image.open(os.path.join(self.root, p)).convert("RGB") for p in paths]

    def user_message(self, row):
        content = [{"type": "image", "image": img} for img in self.images(row)]
        content.append({"type": "text", "text": self.prompt(row)})
        return {"role": "user", "content": content}

    def __getitem__(self, i):
        row = self.rows[i]
        return {"messages": [
            self.user_message(row),
            {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
        ]}

    def to_hf_dataset(self):
        """datasets.Dataset view for SFTTrainer (trl >= 1.x rejects other types).
        Stores only sample indices; images are still opened lazily when a sample is read."""
        from datasets import Dataset

        def transform(batch):
            return {"messages": [self[i]["messages"] for i in batch["idx"]]}

        return Dataset.from_dict({"idx": list(range(len(self)))}).with_transform(transform)
