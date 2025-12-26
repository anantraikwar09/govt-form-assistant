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

# --- STEP 1: SMARTER EXTRACTION LOGIC (PRIORITIZING ENGLISH TO AVOID TRANSLATION ERRORS) ---
def extract_fields(text):
    # Numbers (Very reliable regex)
    pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text.upper())
    aadhaar = re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)
    dob = re.search(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', text)

    # Nuclear Name Extraction: Search & Destroy
    garbage_keywords = [
        'GOVERNMENT', 'INDIA', 'INCOME', 'TAX', 'DEPARTMENT', 'भारत', 'सरकार', 
        'आयकर', 'विभाग', 'CARD', 'MALE', 'FEMALE', 'FATHER', 'ACCOUNT', 'NUMBER',
        'PERMANENT', 'SIGNATURE', 'NOT', 'VALID', 'FOR', 'TRAVEL', 'UNIQUE', 'IDENTIFICATION'
    ]
    
    lines = text.split('\n')
    name = ""
    for line in lines:
        line = line.strip()
        
        # NEW RULE: Skip lines containing Hindi characters to avoid "Anant" -> "Infinite" translation
        if re.search(r'[\u0900-\u097F]', line): 
            continue 

        # Rule 1: No numbers in a name
        if any(char.isdigit() for char in line): continue
        # Rule 2: Must be longer than 3 chars
        if len(line) < 4: continue
        # Rule 3: Skip lines with "Government" or "Department"
        if any(word in line.upper() for word in garbage_keywords): continue
        
        # If it passes, it's the English version of the name
        name = line
        break

    return {
        "PAN": pan.group(0) if pan else "",
        "Aadhaar": aadhaar.group(0) if aadhaar else "",
        "DOB": dob.group(0) if dob else "",
        "Name": name
    }

# --- STEP 2: PDF GENERATION ---
def create_pdf(name, pan, aadhaar, dob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Smart Form Application", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Full Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"PAN Number: {pan}", ln=True)
    pdf.cell(200, 10, txt=f"Aadhaar Number: {aadhaar}", ln=True)
    pdf.cell(200, 10, txt=f"Date of Birth: {dob}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- STEP 3: UI SETTINGS ---
st.set_page_config(page_title="AI Citizen Assistant", layout="centered")
st.title("🇮🇳 AI Form Filler (OCR + Voice)")

uploaded_file = st.file_uploader("Upload ID Card", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Image Preprocessing with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding: Makes text black and background white
    processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    st.image(img, caption="Original Document", width=500)
    
    # 2. Extract Data
    with st.spinner("Analyzing text..."):
        # We extract both but our filter will now pick the English line for the name
        raw_text = pytesseract.image_to_string(processed_img, lang='eng+hin', config='--psm 3')
        data = extract_fields(raw_text)
        
        # Translation removed to prevent semantic errors like Anant -> Infinite
        final_name = data["Name"]

    # 3. User Interface Form
    st.subheader("📝 Verify Extracted Details")
    col1, col2 = st.columns(2)
    with col1:
        f_name = st.text_input("Full Name", value=final_name)
        f_dob = st.text_input("Date of Birth", value=data["DOB"])
    with col2:
        f_pan = st.text_input("PAN Number", value=data["PAN"])
        f_aadhaar = st.text_input("Aadhaar Number", value=data["Aadhaar"])

    # 4. Voice Input for correction
    st.subheader("🎙️ Voice Correction")
    st.info("If the name is wrong, speak it here:")
    audio = mic_recorder(start_prompt="⏺️ Record Name", stop_prompt="⏹️ Stop", key='recorder')
    
    if audio:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio['bytes'])) as source:
            try:
                rec_audio = r.record(source)
                voice_result = r.recognize_google(rec_audio)
                st.success(f"Captured: {voice_result}")
                st.write("Tip: Manually update the Name box with this result.")
            except:
                st.error("Audio was not clear.")

    # 5. Export
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame({"Field": ["Name", "DOB", "PAN", "Aadhaar"], "Value": [f_name, f_dob, f_pan, f_aadhaar]})
        st.download_button("💾 Download CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
    with c2:
        pdf_bytes = create_pdf(f_name, f_pan, f_aadhaar, f_dob)
        st.download_button("📄 Download PDF", pdf_bytes, "application.pdf", "application/pdf")