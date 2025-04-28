import streamlit as st
import pickle
import string

# ======================== Load Model and Vectorizer ========================
@st.cache_resource
def load_model():
    try:
        with open('logistic_regression.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        with open('feature_extraction.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)

        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model or vectorizer: {e}")
        return None, None

# ======================== Preprocessing ========================
def clean_text(text):
    return ''.join([char.lower() for char in text if char not in string.punctuation])

# ======================== Prediction Function ========================
def predict_message(model, vectorizer, msg):
    cleaned = clean_text(msg)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"

# ======================== Streamlit UI ========================
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Message Detector")
st.markdown("Enter a message below and check whether it's **Spam** or **Not Spam**.")

model, vectorizer = load_model()

user_input = st.text_area("✉️ Type your message here:", height=150)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid message.")
    elif model is None or vectorizer is None:
        st.error("❌ Model or vectorizer not loaded.")
    else:
        with st.spinner("Predicting..."):
            result = predict_message(model, vectorizer, user_input)
        st.success(f"🧠 Prediction: **{result}**")

