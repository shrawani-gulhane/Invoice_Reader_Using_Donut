def classify_expense(text):
    text = text.lower()

    if "uber" in text:
        return "Travel"
    elif "restaurant" in text:
        return "Meals"
    else:
        return "Other"


def generate_accounting_entry(data):
    return {
        "expense_type": classify_expense(str(data)),
        "amount": data.get("total", ""),
        "account": "Expense Account"
    }
