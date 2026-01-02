import streamlit as st
import pytesseract
from PIL import Image
import re
import pandas as pd
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import cv2
import numpy as np
from fpdf import FPDF
import random
import time
from pdf2image import convert_from_bytes # NEW: For PDF support

# --- STEP 1: ENHANCED EXTRACTION LOGIC ---
def extract_fields(text):
    # 1. Clean the text for easier matching (remove extra noise)
    clean_text = re.sub(r'[:|]', '', text)

    # 2. Aadhaar Logic: Find all sequences of 4-4-4 or 4-4-4-4 digits
    # We find all matches first, then filter them by length
    all_numbers = re.findall(r'\b\d{4}\s\d{4}\s\d{4}(?:\s\d{4})?\b|\b\d{12,16}\b', clean_text)
    
    aadhaar = ""
    for num in all_numbers:
        # Remove spaces to check the true digit count
        digit_only = num.replace(" ", "")
        # If it's exactly 12 digits, it's the Aadhaar. 
        # If it's 16, it's the VID, so we skip it.
        if len(digit_only) == 12:
            aadhaar = num
            break 

    # 3. Standard regex for other IDs
    pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text.upper())
    voter = re.search(r'[A-Z]{3}[0-9]{7}', text.upper())
    dl = re.search(r'[A-Z]{2}[0-9]{2}\s?[0-9]{11}', text.upper())
    dob = re.search(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', text)
    
    gender = ""
    if re.search(r'MALE|M\s?A\s?L\s?E|पुरुष', text, re.IGNORECASE):
        gender = "Male"
    elif re.search(r'FEMALE|F\s?E\s?M\s?A\s?L\s?E|महिला', text, re.IGNORECASE):
        gender = "Female"

    # 4. Name Logic
    garbage_keywords = [
        'GOVERNMENT', 'INDIA', 'INCOME', 'TAX', 'DEPARTMENT', 'CARD', 'FATHER', 
        'NUMBER', 'SIGNATURE', 'UNIQUE', 'DRIVING', 'LICENSE', 'VOTER', 
        'ELECTION', 'COMMISSION', 'EPIC', 'ENROLLMENT', 'ENROLMENT', 'VID', 'VIRTUAL'
    ]
    
    lines = text.split('\n')
    name = ""
    for line in lines:
        line = line.strip()
        # Remove symbols and digits from the line to isolate potential names
        line = re.sub(r'^[:=\-\|_\.\s\d]+|[:=\-\|_\.\s]+$', '', line)
        
        if len(line) < 4 or re.search(r'[\u0900-\u097F]', line): 
            continue
        if any(word in line.upper() for word in garbage_keywords): 
            continue
        if any(char.isdigit() for char in line):
            continue
            
        name = line
        break

    return {
        "PAN": pan.group(0) if pan else "", 
        "Aadhaar": aadhaar, 
        "Voter ID": voter.group(0) if voter else "", 
        "Driving License": dl.group(0) if dl else "", 
        "DOB": dob.group(0) if dob else "", 
        "Gender": gender, 
        "Name": name
    }
# --- STEP 2: PDF GENERATION ---
def create_pdf(name, pan, aadhaar, voter, dl, dob, gender):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "DIGILOCKER VERIFIED APPLICATION", 1, 1, 'C', True)
    pdf.ln(10)
    data = [
        ["Full Name", name], ["DOB", dob], ["Gender", gender], 
        ["Aadhaar", aadhaar], ["PAN", pan], ["Voter ID", voter], ["DL", dl]
    ]
    for row in data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, row[0], 1)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, str(row[1]) if row[1] else "N/A", 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- STEP 3: STREAMLIT UI ---
st.set_page_config(page_title="Citizen Digital Assistant", layout="wide")

# Sidebar for DigiLocker
with st.sidebar:
    st.image("https://www.digilocker.gov.in/assets/img/logo.png", width=150)
    st.title("DigiLocker Access")
    phone = st.text_input("Enter Aadhaar-Linked Mobile", placeholder="9988776655")
    
    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False
    
    if st.button("Get OTP") and len(phone) == 10:
        st.session_state.generated_otp = str(random.randint(1000, 9999))
        st.session_state.otp_sent = True
        st.success(f"SIMULATED OTP: {st.session_state.generated_otp}") 

    if st.session_state.otp_sent:
        user_otp = st.text_input("Enter 4-Digit OTP")
        if st.button("Verify & Fetch"):
            if user_otp == st.session_state.generated_otp:
                st.success("Authenticated with DigiLocker!")
                st.session_state.digi_data = {
                    "Name": "John Doe", 
                    "DOB": "01/01/1990", 
                    "Gender": "Male", 
                    "Aadhaar": "1234 5678 9012",
                    "PAN": "ABCDE1234F",
                    "Voter ID": "XYZ1234567",
                    "Driving License": "DL01 12345678901"
                }
            else:
                st.error("Invalid OTP")

st.title("🇮🇳 AI Form Filler with DigiLocker & PDF Support")

tab1, tab2 = st.tabs(["📤 Upload Document (PDF/Img)", "☁️ DigiLocker Data"])

with tab1:
    uploaded_file = st.file_uploader("Upload ID Card", type=["jpg", "png", "jpeg", "pdf"])
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        
        # LOGIC: Support both PDF and Image
        if uploaded_file.type == "application/pdf":
            with st.spinner("Processing PDF..."):
                images = convert_from_bytes(file_bytes)
                img_pil = images[0] # Using first page
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Image Enhancement for OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        st.image(img, caption="Document Preview", width=400)
        
        if st.button("Extract Data from Upload"):
            with st.spinner("AI Extracting..."):
                raw_text = pytesseract.image_to_string(processed_img, lang='eng+hin')
                st.session_state.final_data = extract_fields(raw_text)

with tab2:
    if "digi_data" in st.session_state:
        st.write("✅ Data imported from DigiLocker Cloud")
        if st.button("Use DigiLocker Details"):
            st.session_state.final_data = st.session_state.digi_data
    else:
        st.warning("Please authenticate via Sidebar to use DigiLocker.")

# Final Form Rendering
if "final_data" in st.session_state:
    data = st.session_state.final_data
    st.subheader("📝 Final Application Form")
    col1, col2 = st.columns(2)
    with col1:
        f_name = st.text_input("Name", value=data.get("Name", ""))
        f_dob = st.text_input("DOB", value=data.get("DOB", ""))
        f_gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                               index=0 if data.get("Gender")=="Male" else 1)
    with col2:
        f_aadhaar = st.text_input("Aadhaar", value=data.get("Aadhaar", ""))
        f_pan = st.text_input("PAN", value=data.get("PAN", ""))
        f_voter = st.text_input("Voter ID", value=data.get("Voter ID", ""))
        f_dl = st.text_input("Driving License", value=data.get("Driving License", ""))

    st.subheader("🎙️ Voice Correction")
    audio = mic_recorder(start_prompt="⏺️ Record Name", stop_prompt="⏹️ Stop", key='recorder', format="wav")
    if audio:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(io.BytesIO(audio['bytes'])) as source:
                r.adjust_for_ambient_noise(source)
                recorded_audio = r.record(source)
                v_name = r.recognize_google(recorded_audio)
                st.success(f"Detected: {v_name}")
        except: st.error("Audio unclear.")

    if st.button("Generate Final PDF"):
        pdf_bytes = create_pdf(f_name, f_pan, f_aadhaar, f_voter, f_dl, f_dob, f_gender)
        st.download_button("📄 Download Document", pdf_bytes, "verified_app.pdf")
