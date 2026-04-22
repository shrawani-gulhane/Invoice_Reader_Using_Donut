# 🧾 Invoice Reader Using Donut (Document AI System)

A production-style Document AI system for extracting structured information from invoice images using transformer-based vision-language models. This project goes beyond basic OCR by combining deep learning, evaluation pipelines, and business logic to automate invoice understanding at scale.



## 🚀 Project Overview

Manual invoice processing is time-consuming, error-prone, and not scalable. This project builds an end-to-end pipeline that:

* Extracts structured fields from invoice images using Donut (Vision Encoder-Decoder)
* Evaluates model performance with field-level metrics
* Applies business logic for expense classification
* Generates accounting-ready outputs
* Provides a Streamlit interface for real-time interaction



## 🧠 Key Features

* 🔍 Invoice Field Extraction (No OCR required)
* 🧾 Structured JSON outputs (invoice number, date, total, vendor, etc.)
* 📊 Field-level evaluation (accuracy, error analysis)
* 🧪 Template-aware dataset processing (10K+ invoices)
* 💼 Business layer for expense categorization
* 🌐 Streamlit app for demo and visualization
* ⚡ Modular, scalable, and production-ready design



## 🏗️ System Architecture

```text
Invoice Images (FATURA Dataset)
        │
        ▼
Data Preprocessing (BIO → Structured JSON)
        │
        ▼
Donut Model (Vision Encoder-Decoder)
        │
        ▼
Prediction (Structured Output)
        │
        ▼
Evaluation (Field Accuracy)
        │
        ▼
Postprocessing (Business Logic)
        │
        ▼
Streamlit App (User Interface)
```



## 📂 Project Structure

```text
Invoice_Reader_Using_Donut/
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   └── processed/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── src/
│   ├── data/
│   │   ├── prepare_donut_data.py
│   │   └── split_data.py
│   ├── models/
│   │   ├── invoice_dataset.py
│   │   ├── train_donut.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── postprocessing/
│   │   └── postprocess.py
│   └── app/
│       └── streamlit_app.py
│
├── checkpoints/
├── outputs/
├── requirements.txt
└── README.md
```



## 📊 Dataset

* 🧾 ~10,000+ invoice images
* 🧩 Multiple invoice templates
* 🏷️ Token-level annotations (BIO tagging format)

### Data Transformation

The dataset is converted from:

```
Token-level (words + ner_tags)
```

➡️ into:

```json
{
  "image": "invoice_001.jpg",
  "ground_truth": {
    "invoice_no": "...",
    "date": "...",
    "total": "...",
    "vendor": "..."
  }
}
```



## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Invoice_Reader_Using_Donut.git
cd Invoice_Reader_Using_Donut
```



### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔄 Pipeline Execution

### Step 1: Prepare dataset

```bash
python src/data/prepare_donut_data.py
```

---

### Step 2: Split dataset

```bash
python src/data/split_data.py
```

---

### Step 3: Train Donut model

```bash
python src/models/train_donut.py
```

---

### Step 4: Run inference

```bash
python src/models/predict.py
```

---

### Step 5: Launch Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

---

## 📈 Evaluation

The model is evaluated using:

* Field-level accuracy
* Exact match comparison
* Error inspection per field

Example:

```
Invoice Number → 96%
Date → 91%
Total → 94%
```


## 💼 Business Layer

Postprocessing logic converts extracted data into actionable insights:

* Expense classification (Travel, Meals, etc.)
* Accounting-ready structured entries

Example output:

```json
{
  "expense_type": "Travel",
  "amount": 120.50,
  "account": "Transportation Expense"
}
```


## 🧪 Future Enhancements
* 🔁 Donut vs LayoutLMv3 model comparison
* 📊 Template-based generalization testing
* ⚡ Inference optimization (latency benchmarking)
* 🔍 Explainability using SHAP/LIME
* ☁️ Deployment with API endpoints



## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Donut (Vision Encoder-Decoder)
* Pandas / NumPy
* Streamlit
* OpenCV
* Scikit-learn



## 🎯 Results 

* Field-level accuracy: XX%
* Template generalization: XX%
* Processing efficiency improvement: XX%

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repo and submit a PR.

---

## 📬 Contact

For questions or collaboration:

Shrawani Gulhane
📧 shrawanigulhane1902@gmail.com
🔗 https://www.linkedin.com/in/shrawanigulhane/
🌐 https://shrawani-gulhane.github.io/ 

---

## ⭐ If you found this useful

Give this repo a ⭐ to support the project!
