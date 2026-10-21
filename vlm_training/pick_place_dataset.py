"""Load the pick-and-place joint-keyframe dataset (from generate_dataset.py) for Unsloth vision SFT.

In the notebook, replace the LaTeX_OCR dataset cells with:

    from pick_place_dataset import PickPlaceConversations
    converted_dataset = PickPlaceConversations("../datasets/pick_place_v1_21-10-2026-11-03-15", split="train")
    val_dataset       = PickPlaceConversations("../datasets/pick_place_v1_21-10-2026-11-03-15", split="val")

Images are opened lazily per sample, so the whole dataset never sits in RAM.
"""
import os
import json

from PIL import Image


class PickPlaceConversations:
    def __init__(self, root, split="train", cams=None):
        self.root = root
        with open(os.path.join(root, f"{split}.jsonl")) as f:
            self.rows = [json.loads(line) for line in f]
        self.cams = cams  # e.g. ["top"] to train on a subset of the saved cameras

    def __len__(self):
        return len(self.rows)

    def images(self, row):
        paths = row["images"]
        if self.cams is not None:
            paths = [p for p in paths if any(p.endswith(f"_{c}.jpg") for c in self.cams)]
        return [Image.open(os.path.join(self.root, p)).convert("RGB") for p in paths]

    def user_message(self, row):
        content = [{"type": "image", "image": img} for img in self.images(row)]
        content.append({"type": "text", "text": row["prompt"]})
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
