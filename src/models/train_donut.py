import os
import torch
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from src.models.invoice_dataset import InvoiceDataset

# =========================
# PATHS
# =========================
TRAIN_JSON = "data/processed/train_random.json"
VAL_JSON = "data/processed/val_random.json"
IMAGE_DIR = r"C:\Users\sagulhan\Downloads\FATURA\images"
CHECKPOINT_DIR = "checkpoints/donut_resume"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# LOAD PROCESSOR + MODEL
# =========================
if os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
    print("🔁 Resuming from checkpoint...")
    processor = DonutProcessor.from_pretrained(CHECKPOINT_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR)

    if os.path.exists(os.path.join(CHECKPOINT_DIR, "epoch.txt")):
        start_epoch = int(open(os.path.join(CHECKPOINT_DIR, "epoch.txt")).read())
    else:
        start_epoch = 0
else:
    print("🆕 Starting fresh training...")
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base",
        low_cpu_mem_usage=True
    )

    # Add special tokens only on fresh start
    special_tokens = ["<s_invoice>", "</s_invoice>"]
    processor.tokenizer.add_tokens(special_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_invoice>")
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids("</s_invoice>")

    start_epoch = 0

# =========================
# DATASET
# =========================
train_dataset = InvoiceDataset(TRAIN_JSON, IMAGE_DIR, processor)
val_dataset = InvoiceDataset(VAL_JSON, IMAGE_DIR, processor)

# 🔹 TEMP (remove later)
train_dataset.data = train_dataset.data[:200]
val_dataset.data = val_dataset.data[:50]

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)

# =========================
# OPTIMIZER
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# =========================
# TRAINING
# =========================
EPOCHS = 10

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # =========================
    # SAVE CHECKPOINT
    # =========================
    model.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)

    with open(os.path.join(CHECKPOINT_DIR, "epoch.txt"), "w") as f:
        f.write(str(epoch + 1))

    print(f"✅ Saved checkpoint at epoch {epoch+1}")