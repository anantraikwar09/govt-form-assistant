import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import re
import pandas as pd
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import cv2
import numpy as np
from fpdf import FPDF

# --- STEP 1: EXTRACTION LOGIC ---
def extract_fields(text):
    # Numbers
    pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text.upper())
    aadhaar = re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)
    dob = re.search(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', text)
    
    # Gender Logic
    gender = ""
    if re.search(r'MALE|M\s?A\s?L\s?E|पुरुष', text, re.IGNORECASE):
        gender = "Male"
    elif re.search(r'FEMALE|F\s?E\s?M\s?A\s?L\s?E|महिला', text, re.IGNORECASE):
        gender = "Female"

    # Name Extraction (English Script only)
    garbage_keywords = ['GOVERNMENT', 'INDIA', 'INCOME', 'TAX', 'DEPARTMENT', 'CARD', 'FATHER', 'NUMBER', 'SIGNATURE', 'UNIQUE']
    lines = text.split('\n')
    name = ""
    for line in lines:
        line = line.strip()
        if re.search(r'[\u0900-\u097F]', line): continue 
        if any(char.isdigit() for char in line): continue
        if len(line) < 4: continue
        if any(word in line.upper() for word in garbage_keywords): continue
        name = line
        break

    return {
        "PAN": pan.group(0) if pan else "",
        "Aadhaar": aadhaar.group(0) if aadhaar else "",
        "DOB": dob.group(0) if dob else "",
        "Gender": gender,
        "Name": name
    }

# --- STEP 2: ENHANCED PDF GENERATION ---
def create_pdf(name, pan, aadhaar, dob, gender):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "GOVERNMENT SERVICE APPLICATION FORM", 1, 1, 'C', True)
    pdf.ln(10)
    
    # Content Table
    pdf.set_font("Arial", 'B', 12)
    data = [
        ["Full Name", name],
        ["Date of Birth", dob],
        ["Gender", gender],
        ["Aadhaar Number", aadhaar],
        ["PAN Number", pan]
    ]
    
    for row in data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, row[0], 1)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, row[1], 1, 1)
        
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Declaration: I hereby declare that the information provided is true to my knowledge.", 0, 1)
    pdf.ln(10)
    pdf.cell(0, 10, "Signature: __________________________", 0, 1, 'R')
    
    return pdf.output(dest='S').encode('latin-1')

# --- STEP 3: UI ---
st.set_page_config(page_title="AI Citizen Assistant", layout="wide")
st.title("🇮🇳 AI Form Filler & PDF Generator")

uploaded_file = st.file_uploader("Upload ID Card (Aadhaar/PAN)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Processing
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    col_img, col_form = st.columns([1, 1])
    
    with col_img:
        st.image(img, caption="Uploaded Document", use_container_width=True)
        
    with col_form:
        with st.spinner("Extracting data..."):
            raw_text = pytesseract.image_to_string(processed_img, lang='eng+hin', config='--psm 3')
            data = extract_fields(raw_text)

        st.subheader("📝 Verify Details")
        f_name = st.text_input("Name", value=data["Name"])
        f_dob = st.text_input("DOB", value=data["DOB"])
        f_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if data["Gender"] == "Male" else 1 if data["Gender"] == "Female" else 2)
        f_pan = st.text_input("PAN", value=data["PAN"])
        f_aadhaar = st.text_input("Aadhaar", value=data["Aadhaar"])

        # --- UPDATED VOICE CORRECTION SECTION ---
        st.subheader("🎙️ Voice Correction")
        st.info("Record your name if the text extraction was incorrect.")
        audio = mic_recorder(
            start_prompt="⏺️ Record Name", 
            stop_prompt="⏹️ Stop", 
            key='recorder',
            format="wav"  # Forces WAV format to prevent ValueError
        )

        if audio:
            r = sr.Recognizer()
            try:
                # Wrap bytes in BytesIO so sr.AudioFile can read them
                audio_file = io.BytesIO(audio['bytes'])
                with sr.AudioFile(audio_file) as source:
                    # Help Google recognize better by adjusting for silence/noise
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    recorded_audio = r.record(source)
                    
                    # Call Google Speech API
                    v_name = r.recognize_google(recorded_audio)
                    st.success(f"Detected: {v_name}")
                    st.caption("You can now update the Name field with the detected text.")
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio.")
            except sr.RequestError:
                st.error("Could not request results from Google Speech Recognition service.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        st.divider()
        if st.button("Generate Documents"):
            st.balloons()
            
        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame({"Field": ["Name", "DOB", "Gender", "PAN", "Aadhaar"], "Value": [f_name, f_dob, f_gender, f_pan, f_aadhaar]})
            st.download_button("💾 CSV", df.to_csv(index=False).encode('utf-8'), "data.csv")
        with c2:
            pdf_bytes = create_pdf(f_name, f_pan, f_aadhaar, f_dob, f_gender)
            st.download_button("📄 PDF Form", pdf_bytes, "application.pdf")
