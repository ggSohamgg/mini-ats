
# 📄 Mini ATS – Resume vs Job Description Matcher

An end-to-end Machine Learning system that evaluates the compatibility between a candidate’s resume and a job description using NLP and classification techniques.

---

## 🚀 Features

- 🔍 Multi-class classification: **Good Fit / Potential Fit / No Fit**
- 🧠 TF-IDF based feature extraction
- ⚖️ Class imbalance handling with weighted Logistic Regression
- 📊 Model evaluation with detailed metrics
- 💡 Explainable predictions with confidence scores
- 🌐 Deployable using Streamlit / Gradio

---

## 📊 Model Performance



Accuracy: 0.5981

``
           precision    recall  f1-score   support

 Good Fit       0.54      0.78      0.64       309
   No Fit       0.77      0.46      0.58       629
``

Potential Fit       0.50      0.69      0.58       311

``
 accuracy                           0.60      1249
macro avg       0.60      0.64      0.60      1249
``

weighted avg       0.65      0.60      0.59      1249

``

📌 Note:
- Higher recall for **Good Fit** and **Potential Fit** improves real-world ATS usefulness
- Slight trade-off in overall accuracy due to class balancing

---

## 📚 Dataset

Trained on:

👉 https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit

- Contains resume–job description pairs
- Labels: `Good Fit`, `Potential Fit`, `No Fit`
- ~6K samples

---

## 🧠 ML Pipeline

```mermaid
flowchart LR
    A[Resume Text] --> C[Text Cleaning]
    B[Job Description] --> C
    C --> D[Combine with SEP Token]
    D --> E[TF-IDF Vectorization]
    E --> F[Logistic Regression Model]
    F --> G[Prediction]
    G --> H[Class Output]
    G --> I[Confidence Scores]
````

---

## ⚙️ Tech Stack

* Python
* scikit-learn
* TF-IDF (NLP)
* Logistic Regression
* Streamlit / Gradio
* Hugging Face Datasets

---

## 🏗️ Project Structure

``
mini-ats/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone repo

```
git clone https://github.com/YOUR_USERNAME/mini-ats.git
cd mini-ats
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run app

```
streamlit run app.py
```

---

## 💡 Example Output

```
Prediction: Potential Fit

Confidence:
Good Fit: 0.29
No Fit: 0.33
Potential Fit: 0.38
```

