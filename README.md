# 📩 Spam Email Classifier

This project is a Machine Learning-based **Spam Email Classifier** that detects whether a given message is **Spam** or **Not Spam**.  
It uses **Logistic Regression**, **TF-IDF Vectorization**, and is deployed via a **Streamlit** web app for real-time prediction.

---

## 🧠 Problem Statement

Spam emails are a major problem — they waste time, carry scams, and pose cybersecurity threats.  
This project aims to build an automated system that:
- Analyzes email messages,
- Extracts features from the text,
- Classifies them as **Spam** or **Not Spam** with high accuracy.

---

## 🏗️ Architecture Diagram

![Architecture](architecture-diagram.png)  
*(Upload the architecture image as `architecture-diagram.png` into your repo)*

---

## ⚙️ Technologies Used

- **Python**
- **Pandas** – for data processing
- **Scikit-learn** – for ML model and vectorizer
- **Streamlit** – for building web app
- **Pickle** – for saving model and vectorizer
- **GitHub + Streamlit Cloud** – for deployment

---

## 📂 Project Structure

spam-email-classifier/ │ ├── accuracy.py # Model training and evaluation script ├── app.py # Streamlit web app code ├── mail_data.csv # Dataset (SMS spam collection) ├── logistic_regression.pkl # Trained ML model ├── feature_extraction.pkl # TF-IDF vectorizer ├── requirements.txt # Required libraries for deployment └── README.md # Project documentation


---

## 🧪 Model Details

- **Model**: Logistic Regression
- **Vectorizer**: TF-IDF
- **Accuracy**: ~96.5%
- **Evaluation**:
  - Confusion Matrix
  - Classification Report

---

## 🚀 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

Install dependencies


pip install -r requirements.txt
(Optional) Train the model


python accuracy.py
Run the Streamlit app


streamlit run app.py
Try Messages like:

"Congratulations! You've won a free iPhone" → 🚨 Spam

"Let's meet at 5 PM for coffee" → ✅ Not Spam

🌐 Live Demo
🔗 Click here to try the Live App
(Replace with your actual deployed Streamlit Cloud link)

📈 Results
Achieved high accuracy using simple yet effective ML techniques

Real-time predictions via web UI

Fast, lightweight, and easy to use

🔮 Future Improvements
Use deep learning models (LSTM) for contextual understanding

Add multilingual message support

Improve UI with detailed feedback and explanations

Auto-retraining pipeline with feedback loop

🙌 Acknowledgements
UCI SMS Spam Dataset

Streamlit.io for web deployment

👨‍💻 Created by: Karthik Aljapur
If you found this helpful, give the project a ⭐ on GitHub!
