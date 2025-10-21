import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os


# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="💊 স্বাস্থ্য সহায়ক স্মার্ট চ্যাটবট",
    layout="wide",
    page_icon="💊"
)

# ---------------- Header ----------------
st.title("💊 স্বাস্থ্য সহায়ক স্মার্ট চ্যাটবট")
st.subheader("রোগীর উপসর্গ অনুযায়ী স্বয়ংক্রিয় স্বাস্থ্য পরামর্শ")
st.markdown("---")

# ---------------- Chat Log Storage ----------------
if not os.path.exists("chat_logs"):
    os.makedirs("chat_logs")

def save_chat_log(patient_question, bot_answer):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df = pd.DataFrame([[patient_question, bot_answer, now]], columns=["রোগীর প্রশ্ন", "চ্যাটবট উত্তর", "সময়"])
    file_path = f"chat_logs/chat_log_{now}.csv"
    df.to_csv(file_path, index=False)
    return file_path

# ---------------- Load Predefined Symptoms ----------------
symptom_df = pd.read_csv("symptom_responses.csv")
questions = symptom_df["উপসর্গ/প্রশ্ন"].tolist()
answers = symptom_df["উত্তর"].tolist()

# ---------------- NLP Function ----------------
vectorizer = TfidfVectorizer()
vectorizer.fit(questions)

def get_bot_response(user_question):
    all_vectors = vectorizer.transform([user_question] + questions)
    similarity = cosine_similarity(all_vectors[0:1], all_vectors[1:]).flatten()
    index = similarity.argmax()
    if similarity[index] < 0.2:  # threshold for unknown
        return "দুঃখিত, আমি আপনার উপসর্গ ঠিকভাবে বুঝতে পারছি না। বিস্তারিত বলুন বা ডাক্তার দেখুন।"
    else:
        return answers[index]

# ---------------- Chat Input ----------------
patient_question = st.text_area("আপনার উপসর্গ বা প্রশ্ন লিখুন (বাংলায়)", placeholder="উদাহরণ: আমার মাথা ব্যথা করছে")
if st.button("প্রেরণ করুন"):
    if patient_question.strip() == "":
        st.warning("⚠️ দয়া করে আপনার প্রশ্ন লিখুন।")
    else:
        bot_answer = get_bot_response(patient_question)
        st.markdown(f"**চ্যাটবট উত্তর:** {bot_answer}")
        log_file = save_chat_log(patient_question, bot_answer)
        st.success("✅ চ্যাট লগ সংরক্ষণ করা হয়েছে।")
        with open(log_file, "rb") as f:
            st.download_button(
                "📥 চ্যাট লগ ডাউনলোড করুন",
                data=f,
                file_name=f"chat_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                mime="text/csv"
            )

# ---------------- Appointment Booking ----------------
st.markdown("---")
st.subheader("ডাক্তার অ্যাপয়েন্টমেন্ট বুক করুন")
appointment_name = st.text_input("আপনার নাম")
appointment_date = st.date_input("অ্যাপয়েন্টমেন্টের তারিখ")
if st.button("অ্যাপয়েন্টমেন্ট বুক করুন"):
    if appointment_name.strip() == "":
        st.warning("⚠️ দয়া করে আপনার নাম লিখুন।")
    else:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_appt = pd.DataFrame([[appointment_name, str(appointment_date), now]], columns=["নাম", "অ্যাপয়েন্টমেন্ট তারিখ", "রেজিস্টার সময়"])
        appt_file = f"chat_logs/appointment_{now}.csv"
        df_appt.to_csv(appt_file, index=False)
        st.success(f"✅ অ্যাপয়েন্টমেন্ট সফলভাবে বুক হয়েছে। ({appointment_date})")
        with open(appt_file, "rb") as f:
            st.download_button(
                "📥 অ্যাপয়েন্টমেন্ট ডাউনলোড করুন",
                data=f,
                file_name=f"appointment_{now}.csv",
                mime="text/csv"
            )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 Md Junayed Bin Karim</p>", unsafe_allow_html=True)
