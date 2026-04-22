import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image


def predict(image_path):
    processor = DonutProcessor.from_pretrained("checkpoints/donut_model")
    model = VisionEncoderDecoderModel.from_pretrained("checkpoints/donut_model")

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(pixel_values)

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print(decoded)


if __name__ == "__main__":
    predict("data/raw/images/sample.jpg")
