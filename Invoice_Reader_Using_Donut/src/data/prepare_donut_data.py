import os
import json
from tqdm import tqdm

# 🔹 UPDATE THIS PATH (VERY IMPORTANT)
ANNOTATION_DIR = r"C:\Users\shraw\Downloads\FATURA\invoices_dataset_final\Annotations\layoutlm_HF_format"

# 🔹 OUTPUT PATH
OUTPUT_PATH = r"C:\Users\shraw\OneDrive\Documents\Invoice_Reader_Using_Donut\data\processed\donut_all.json"


def extract_fields(words, ner_tags):
    """
    Hybrid extraction:
    - Use tags to group
    - Use keywords to classify
    """

    # Step 1: group text by tag
    tag_groups = {}

    for word, tag in zip(words, ner_tags):
        tag_groups.setdefault(tag, []).append(word)

    grouped_text = {
        tag: " ".join(words_list)
        for tag, words_list in tag_groups.items()
    }

    # Step 2: classify using keywords
    result = {}

    for text in grouped_text.values():
        t = text.lower()

        # Invoice number
        if "invoice" in t and "#" in t:
            result["invoice_no"] = text

        # Date
        # Due date
        elif "due date" in t:
           result["due_date"] = text

        # Invoice date
        elif "date" in t:
           result["date"] = text

        # Total
        elif "total" in t:
            result["total_amount"] = text

        # Buyer
        elif "bill to" in t or "buyer" in t:
            result["buyer"] = text

    return result


def convert_fatura_to_donut():
    dataset = []

    files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} annotation files...")

    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(ANNOTATION_DIR, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 🔹 Extract required fields
        image_name = data["path"]
        words = data["words"]
        ner_tags = data["ner_tags"]

        ground_truth = extract_fields(words, ner_tags)

        # 🔹 Skip empty samples
        if len(ground_truth) == 0:
            continue

        dataset.append({
            "image": image_name,
            "ground_truth": ground_truth
        })

        # 🔹 DEBUG FIRST SAMPLE
        if i == 0:
            print("\n--- DEBUG SAMPLE ---")
            print("WORDS:", words[:20])
            print("TAGS:", ner_tags[:20])
            print("CLEANED OUTPUT:", ground_truth)
            print("--------------------\n")

    print(f"Total usable samples: {len(dataset)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_fatura_to_donut()