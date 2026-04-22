import json


def evaluate(predictions, ground_truths):
    correct = 0
    total = 0

    for pred, gt in zip(predictions, ground_truths):
        for key in gt:
            total += 1
            if key in pred and pred[key] == gt[key]:
                correct += 1

    accuracy = correct / total
    print(f"Field Accuracy: {accuracy:.4f}")
