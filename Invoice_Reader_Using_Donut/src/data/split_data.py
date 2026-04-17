import os
import re
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

INPUT_PATH = r"C:\Users\shraw\OneDrive\Documents\Invoice_Reader_Using_Donut\data\processed\donut_all.json"
OUTPUT_DIR = r"C:\Users\shraw\OneDrive\Documents\Invoice_Reader_Using_Donut\data\processed"

RANDOM_SEED = 42


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_dataset():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def random_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."

    train_data, temp_data = train_test_split(
        dataset,
        test_size=(1 - train_ratio),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    val_relative_ratio = val_ratio / (val_ratio + test_ratio)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_relative_ratio),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    return train_data, val_data, test_data


def extract_template_name(image_name):
    """
    Expected pattern examples:
    Template10_Instance0.jpg
    Template3_Instance145.png

    Returns:
        Template10
    or None if pattern not found
    """
    match = re.search(r"(Template\d+)_Instance\d+", image_name, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def template_aware_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."

    grouped = defaultdict(list)
    no_template = []

    for item in dataset:
        image_name = item["image"]
        template_name = extract_template_name(image_name)

        if template_name is None:
            no_template.append(item)
        else:
            grouped[template_name].append(item)

    template_names = list(grouped.keys())
    template_names.sort()
    random.seed(RANDOM_SEED)
    random.shuffle(template_names)

    total_templates = len(template_names)
    train_count = int(total_templates * train_ratio)
    val_count = int(total_templates * val_ratio)

    train_templates = set(template_names[:train_count])
    val_templates = set(template_names[train_count:train_count + val_count])
    test_templates = set(template_names[train_count + val_count:])

    train_data, val_data, test_data = [], [], []

    for template_name, items in grouped.items():
        if template_name in train_templates:
            train_data.extend(items)
        elif template_name in val_templates:
            val_data.extend(items)
        else:
            test_data.extend(items)

    # If some files do not follow template naming, put them in train and report
    train_data.extend(no_template)

    return train_data, val_data, test_data, train_templates, val_templates, test_templates, no_template


def print_split_stats(name, train_data, val_data, test_data):
    total = len(train_data) + len(val_data) + len(test_data)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Train: {len(train_data)} ({len(train_data)/total:.2%})")
    print(f"Val:   {len(val_data)} ({len(val_data)/total:.2%})")
    print(f"Test:  {len(test_data)} ({len(test_data)/total:.2%})")
    print(f"Total: {total}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples.")

    # ---------- Random split ----------
    rand_train, rand_val, rand_test = random_split(dataset)

    save_json(rand_train, os.path.join(OUTPUT_DIR, "train_random.json"))
    save_json(rand_val, os.path.join(OUTPUT_DIR, "val_random.json"))
    save_json(rand_test, os.path.join(OUTPUT_DIR, "test_random.json"))

    print_split_stats("Random Split", rand_train, rand_val, rand_test)

    # ---------- Template-aware split ----------
    tmpl_train, tmpl_val, tmpl_test, train_templates, val_templates, test_templates, no_template = template_aware_split(dataset)

    save_json(tmpl_train, os.path.join(OUTPUT_DIR, "train_template.json"))
    save_json(tmpl_val, os.path.join(OUTPUT_DIR, "val_template.json"))
    save_json(tmpl_test, os.path.join(OUTPUT_DIR, "test_template.json"))

    split_metadata = {
        "train_templates": sorted(list(train_templates)),
        "val_templates": sorted(list(val_templates)),
        "test_templates": sorted(list(test_templates)),
        "no_template_match_count": len(no_template)
    }

    save_json(split_metadata, os.path.join(OUTPUT_DIR, "template_split_metadata.json"))

    print_split_stats("Template-Aware Split", tmpl_train, tmpl_val, tmpl_test)
    print(f"Templates in train: {len(train_templates)}")
    print(f"Templates in val:   {len(val_templates)}")
    print(f"Templates in test:  {len(test_templates)}")
    print(f"Files without template pattern: {len(no_template)}")


if __name__ == "__main__":
    main()