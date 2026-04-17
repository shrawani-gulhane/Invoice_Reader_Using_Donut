import os
import json
from collections import defaultdict
from tqdm import tqdm


ANNOTATION_DIR = r"C:\Users\shraw\Downloads\FATURA\invoices_dataset_final\Annotations\layoutlm_HF_format"
OUTPUT_PATH = r"C:\Users\shraw\OneDrive\Documents\Invoice_Reader_Using_Donut\data\processed\donut_all.json"


def extract_fields(words, ner_tags):
    from collections import defaultdict

    field_map = defaultdict(list)

    for word, tag in zip(words, ner_tags):
        if tag != 0:  # ignore background
            field_map[tag].append(word)

    # Convert tag IDs to meaningful names (we define manually)
    tag_to_field = {
        1: "invoice_total_label",
        3: "date",
        5: "buyer_info",
        6: "seller_name",
        10: "logo",
        11: "gst_label",
        12: "invoice_label",
        13: "important_value"
    }

    structured = {}

    for tag, words_list in field_map.items():
        field_name = tag_to_field.get(tag, f"field_{tag}")
        structured[field_name] = " ".join(words_list)

    return structured


def convert_fatura_to_donut():
    dataset = []

    files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")]

    print(f"Found {len(files)} annotation files...")

    for file in tqdm(files):
        file_path = os.path.join(ANNOTATION_DIR, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_name = data["path"]
        words = data["words"]
        ner_tags = data["ner_tags"]

        ground_truth = extract_fields(words, ner_tags)

        sample = {
            "image": image_name,
            "ground_truth": ground_truth
        }

        dataset.append(sample)

    print(f"Total samples: {len(dataset)}")

    # Save dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_fatura_to_donut()