import os
os.environ["OMP_NUM_THREADS"] = "2"
import json
from PIL import Image
from torch.utils.data import Dataset


class InvoiceDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, max_length=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")

        target_sequence = "<s_invoice>" + json.dumps(item["ground_truth"]) + "</s_invoice>"

        pixel_values = self.processor(
            image,
            return_tensors="pt",
            size={"height": 960, "width": 720}
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            target_sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }