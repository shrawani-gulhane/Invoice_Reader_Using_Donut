import torch
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from src.models.invoice_dataset import InvoiceDataset

TRAIN_JSON = "data/processed/train_random.json"
VAL_JSON = "data/processed/val_random.json"
IMAGE_DIR = r"C:\Users\shraw\Downloads\FATURA\invoices_dataset_final\images"

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base",
    low_cpu_mem_usage=True
)

special_tokens = ["<s_invoice>", "</s_invoice>"]
processor.tokenizer.add_tokens(special_tokens)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_invoice>")
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids("</s_invoice>")

train_dataset = InvoiceDataset(TRAIN_JSON, IMAGE_DIR, processor)
val_dataset = InvoiceDataset(VAL_JSON, IMAGE_DIR, processor)

# small test run first
train_dataset.data = train_dataset.data[:50]
val_dataset.data = val_dataset.data[:10]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

EPOCHS = 3

for epoch in range(EPOCHS):
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

model.save_pretrained("checkpoints/donut_invoice_model")
processor.save_pretrained("checkpoints/donut_invoice_model")

print("Model saved successfully!")