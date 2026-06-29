"""
VQA dataset that returns (pixel_values, input_ids, attention_mask, answer_id)
and optionally pre-cached teacher features.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


def vqa_collate_fn(batch: list) -> dict:
    """Pad variable-length input_ids / attention_mask to the longest in the batch."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    answer_ids = torch.stack([b["answer_id"] for b in batch])
    question_ids = [b["question_id"] for b in batch]

    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 1  # Florence-2 / BART uses pad_token_id = 1
    input_ids = torch.stack([
        torch.nn.functional.pad(b["input_ids"], (0, max_len - b["input_ids"].shape[0]), value=pad_id)
        for b in batch
    ])
    attention_mask = torch.stack([
        torch.nn.functional.pad(b["attention_mask"], (0, max_len - b["attention_mask"].shape[0]), value=0)
        for b in batch
    ])

    out = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_id": answer_ids,
        "question_id": question_ids,
    }

    if "teacher_feats" in batch[0]:
        n_layers = len(batch[0]["teacher_feats"])
        out["teacher_feats"] = [
            torch.stack([b["teacher_feats"][i] for b in batch]) for i in range(n_layers)
        ]
        out["teacher_attns"] = [
            torch.stack([b["teacher_attns"][i] for b in batch]) for i in range(n_layers)
        ]

    return out


class VQADataset(Dataset):
    def __init__(
        self,
        samples_file: str,
        vocab_file: str,
        image_dir: str,
        processor_name: str = "microsoft/Florence-2-base",
        teacher_cache_file: Optional[str] = None,
    ):
        with open(samples_file) as f:
            self.samples = json.load(f)
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        self.unk_id = self.vocab.get("<unk>", len(self.vocab) - 1)
        self.image_dir = Path(image_dir)

        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            trust_remote_code=True,
            revision="5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
        )

        # Optional: pre-cached teacher features stored as a dict keyed by question_id
        self.teacher_cache = None
        if teacher_cache_file and Path(teacher_cache_file).exists():
            import h5py
            self._h5_file = h5py.File(teacher_cache_file, "r")
            self.teacher_cache = self._h5_file

    def __len__(self) -> int:
        return len(self.samples)

    def _get_answer_id(self, answers: list) -> int:
        # majority vote over 10 annotators
        from collections import Counter
        votes = Counter(a["answer"].lower().strip() for a in answers)
        best_ans = votes.most_common(1)[0][0]
        return self.vocab.get(best_ans, self.unk_id)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.image_dir / f"COCO_val2014_{sample['image_id']:012d}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Florence-2 processor: do NOT pass max_length here.
        # image_seq_length=577, so any max_length<577 would go negative.
        # Padding is handled per-batch in vqa_collate_fn instead.
        task_prompt = "<VQA>"
        text_input = task_prompt + sample["question"]
        inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt",
        )

        answer_id = self._get_answer_id(sample["answers"])
        item = {
            "pixel_values": inputs["pixel_values"].squeeze(0),     # (C, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),           # (T,)
            "attention_mask": inputs["attention_mask"].squeeze(0), # (T,)
            "answer_id": torch.tensor(answer_id, dtype=torch.long),
            "question_id": sample["question_id"],
        }

        if self.teacher_cache is not None:
            qid = str(int(sample["question_id"]))
            try:
                item["teacher_feats"] = [
                    torch.from_numpy(self.teacher_cache[qid]["feats"][str(i)][:])
                    for i in range(len(self.teacher_cache[qid]["feats"]))
                ]
                item["teacher_attns"] = [
                    torch.from_numpy(self.teacher_cache[qid]["attns"][str(i)][:])
                    for i in range(len(self.teacher_cache[qid]["attns"]))
                ]
            except KeyError:
                pass  # sample not in cache; trainer falls back to task loss only

        return item


def build_dataloader(
    samples_file: str,
    vocab_file: str,
    image_dir: str,
    batch_size: int,
    processor_name: str = "microsoft/Florence-2-base",
    teacher_cache_file: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = VQADataset(
        samples_file=samples_file,
        vocab_file=vocab_file,
        image_dir=image_dir,
        processor_name=processor_name,
        teacher_cache_file=teacher_cache_file,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=vqa_collate_fn,
    )


if __name__ == "__main__":
    import yaml

    with open("configs/distil_config.yaml") as f:
        cfg = yaml.safe_load(f)

    loader = build_dataloader(
        samples_file="data/cache/vqa_samples.json",
        vocab_file="data/cache/answer_vocab.json",
        image_dir="data/raw/val2017",
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(loader))
    print("pixel_values:", batch["pixel_values"].shape)
    print("input_ids:", batch["input_ids"].shape)
    print("answer_id:", batch["answer_id"])
    print("Dataset size:", len(loader.dataset))
