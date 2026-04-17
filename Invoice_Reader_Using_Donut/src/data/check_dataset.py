import json

FILE_PATH = r"C:\Users\shraw\OneDrive\Documents\Invoice_Reader_Using_Donut\data\processed\train_random.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}\n")

for i in range(min(3, len(data))):
    print(f"Sample {i+1}")
    print("Image:", data[i]["image"])
    print("Ground truth:", json.dumps(data[i]["ground_truth"], indent=2))
    print("-" * 50)