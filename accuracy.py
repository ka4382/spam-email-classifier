import pandas as pd
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================== 1. Load Dataset ========================
df = pd.read_csv("mail_data.csv", encoding='latin-1')[["Category", "Message"]]
df = df.rename(columns={"Category": "label", "Message": "text"})

# ======================== 2. Preprocessing ========================
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Optional: remove punctuation and lowercase
def clean_text(text):
    return ''.join([char.lower() for char in text if char not in string.punctuation])

df['text'] = df['text'].apply(clean_text)

# ======================== 3. Train-Test Split ========================
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================== 4. TF-IDF Vectorization ========================
vectorizer = TfidfVectorizer(stop_words='english')  # Removed max_features to use full vocab
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======================== 5. Train Logistic Regression with Class Balancing ========================
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

model.fit(X_train_vec, y_train)

# ======================== 6. Evaluation ========================
y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("🧠 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ======================== 7. Save Model and Vectorizer ========================
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_extraction.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved as 'logistic_regression.pkl' and 'feature_extraction.pkl'")
