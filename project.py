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
from pdf2image import convert_from_bytes

# --- STEP 1: UPDATED EXTRACTION LOGIC WITH GEOMETRIC ROI ---
def extract_fields(text, img):
    # Standard clean up
    clean_text = re.sub(r'[:|]', '', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # ID Number Extraction
    all_numbers = re.findall(r'\b\d{4}\s\d{4}\s\d{4}(?:\s\d{4})?\b|\b\d{12,16}\b', clean_text)
    aadhaar = ""
    for num in all_numbers:
        digit_only = num.replace(" ", "")
        if len(digit_only) == 12:
            aadhaar = num
            break 

    pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text.upper())
    voter = re.search(r'[A-Z]{3}[0-9]{7}', text.upper())
    dl = re.search(r'[A-Z]{2}[0-9]{2}\s?[0-9]{11}', text.upper())
    dob = re.search(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', text)
    
    gender = "Male"
    if re.search(r'FEMALE|महिला|F\s?E\s?M\s?A\s?L\s?E', text, re.IGNORECASE):
        gender = "Female"

    # --- UPGRADED: GEOMETRIC ADDRESS EXTRACTION ---
    address = ""
    # Use pytesseract to get detailed data including coordinates
    data = pytesseract.image_to_data(img, lang='eng+hin+mar', output_type=pytesseract.Output.DICT)
    
    start_x, start_y = -1, -1
    addr_starts = ['S/O', 'D/O', 'W/O', 'C/O', 'ADDRESS', 'पता']
    
    # Find the pixel coordinate (x, y) of the starting anchor
    for i, word in enumerate(data['text']):
        if any(marker in word.upper() for marker in addr_starts):
            start_x, start_y = data['left'][i], data['top'][i]
            break
            
    if start_x != -1:
        h, w, _ = img.shape
        # Create a rectangular boundary: 
        # Start at anchor, take 75% of image width (ignores side vertical text)
        # Take 25% of image height (captures roughly 4-5 lines)
        roi_x = max(0, start_x - 5)
        roi_y = max(0, start_y - 5)
        roi_w = int(w * 0.75) 
        roi_h = int(h * 0.25)
        
        # Crop the image to the address block only
        crop_address_img = img[roi_y:min(roi_y+roi_h, h), roi_x:min(roi_x+roi_w, w)]
        
        # OCR on the cropped "clean" box
        address_raw = pytesseract.image_to_string(crop_address_img, lang='eng+hin+mar').strip()
        
        # Remove end markers if caught in the crop (Mobile, Help, etc.)
        address_clean = re.split(r'Mobile|Phone|दूरभाष|Help|www|unique', address_raw, flags=re.IGNORECASE)[0]
        
        # Remove the starting anchor word from the final text
        for m in addr_starts:
            address_clean = re.sub(m, '', address_clean, flags=re.IGNORECASE)
            
        address = re.sub(r'\s+', ' ', address_clean).strip()

    # Name Extraction Logic
    garbage_keywords = ['GOVERNMENT', 'INDIA', 'INCOME', 'TAX', 'DEPARTMENT', 'CARD', 'FATHER', 'NUMBER', 'SIGNATURE', 'UNIQUE', 'DRIVING', 'LICENSE', 'VOTER', 'ELECTION', 'COMMISSION', 'EPIC', 'ENROLLMENT', 'ENROLMENT', 'VID', 'VIRTUAL']
    name = ""
    for line in lines:
        line_clean = re.sub(r'^[:=\-\|_\.\s\d]+|[:=\-\|_\.\s]+$', '', line)
        if len(line_clean) < 4 or re.search(r'[\u0900-\u097F]', line_clean): continue
        if any(word in line_clean.upper() for word in garbage_keywords): continue
        if any(char.isdigit() for char in line_clean): continue
        name = line_clean
        break

    return {
        "PAN": pan.group(0) if pan else "", 
        "Aadhaar": aadhaar, 
        "Voter ID": voter.group(0) if voter else "", 
        "Driving License": dl.group(0) if dl else "", 
        "DOB": dob.group(0) if dob else "", 
        "Gender": gender, 
        "Name": name,
        "Address": address
    }

# --- STEP 2: FACE DETECTION & MASKING ---
def extract_face(image):
    # 1. Load the pre-trained face model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 2. Convert to grayscale and improve contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    
    # 3. Detect faces with tuned parameters for full documents
    # scaleFactor=1.05: Checks for faces at many different sizes (slower but more accurate)
    # minNeighbors=5: Higher value means fewer false positives
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05, 
        minNeighbors=6, 
        minSize=(60, 60)
    )
    
    if len(faces) > 0:
        # 4. Fallback: Find the LARGEST face (usually the main photo)
        # Full Aadhaar cards often have a tiny "ghost" image; we want the big one.
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face
        
        # 5. Add a 20% padding around the face for a better crop
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        
        face_img = image[
            max(0, y - pad_h) : min(image.shape[0], y + h + pad_h), 
            max(0, x - pad_w) : min(image.shape[1], x + w + pad_w)
        ]
        return face_img
        
    return None
def mask_number(number):
    if not number or len(str(number)) < 4: return number
    return "XXXX-XXXX-" + str(number)[-4:]

# --- STEP 3: PDF GENERATION ---
def create_pdf(name, pan, aadhaar, voter, dl, dob, gender, address):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "VERIFIED CITIZEN APPLICATION", 1, 1, 'C', True)
    pdf.ln(10)
    
    data = [
        ["Full Name", name], ["DOB", dob], ["Gender", gender], 
        ["Aadhaar", aadhaar], ["PAN", pan], ["Voter ID", voter], ["DL", dl]
    ]
    for row in data:
        pdf.set_font("Arial", 'B', 12); pdf.cell(60, 10, row[0], 1)
        pdf.set_font("Arial", '', 12); pdf.cell(0, 10, str(row[1]) if row[1] else "N/A", 1, 1)
    
    pdf.set_font("Arial", 'B', 12); pdf.cell(60, 10, "Address", 1)
    pdf.set_font("Arial", '', 10); pdf.multi_cell(0, 10, str(address) if address else "N/A", 1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- STEP 4: STREAMLIT UI ---
st.set_page_config(page_title="Citizen Digital Assistant", layout="wide")

if "final_data" not in st.session_state:
    st.session_state.final_data = None
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False

with st.sidebar:
    st.image("https://www.digilocker.gov.in/assets/img/logo.png", width=150)
    st.title("DigiLocker Access")
    phone = st.text_input("Enter Mobile", placeholder="9988776655")
    
    if st.button("Get OTP") and len(phone) == 10:
        st.session_state.generated_otp = str(random.randint(1000, 9999))
        st.session_state.otp_sent = True
        st.success(f"SIMULATED OTP: {st.session_state.generated_otp}") 

    if st.session_state.otp_sent:
        user_otp = st.text_input("Enter 4-Digit OTP")
        if st.button("Verify & Fetch"):
            if user_otp == st.session_state.generated_otp:
                st.success("Authenticated!")
                st.session_state.final_data = {
                    "Name": "John Doe", "DOB": "01/01/1990", "Gender": "Male", 
                    "Aadhaar": "1234 5678 9012", "PAN": "ABCDE1234F",
                    "Address": "House No 123, Street Name, City, State - 000000"
                }
            else: st.error("Invalid OTP")

st.title("🇮🇳 AI Form Filler (v2.3)")

tab1, tab2 = st.tabs(["📤 Upload ID", "📄 Final Application"])

with tab1:
    uploaded_file = st.file_uploader("Upload ID (PDF/Image)", type=["jpg", "png", "jpeg", "pdf"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(file_bytes)
            img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col_img, col_face = st.columns([2, 1])
        with col_img:
            st.image(img, caption="Document Preview", use_container_width=True)
        with col_face:
            face = extract_face(img)
            if face is not None: st.image(face, caption="Detected Face", width=150)
            else: st.info("No face detected.")

        if st.button("🚀 Run AI Extraction"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🎨 Pre-processing image...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            progress_bar.progress(30)
            time.sleep(0.4)
            
            status_text.text("🔍 OCR Engine Scan...")
            raw_text = pytesseract.image_to_string(processed_img, lang='eng+hin+mar')
            progress_bar.progress(70)
            time.sleep(0.4)
            
            status_text.text("📂 Mapping Address ROI...")
            # MODIFIED: Passing img to extract_fields for ROI processing
            st.session_state.final_data = extract_fields(raw_text, img)
            progress_bar.progress(100)
            
            status_text.text("✅ Extraction Complete!")
            st.balloons()
            st.success("Check 'Final Application' tab.")

with tab2:
    if st.session_state.final_data:
        data = st.session_state.final_data
        st.subheader("📝 Edit & Verify Details")
        
        c1, c2 = st.columns(2)
        with c1:
            f_name = st.text_input("Full Name", data.get("Name", ""))
            f_dob = st.text_input("Date of Birth", data.get("DOB", ""))
            f_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if data.get("Gender")=="Male" else 1)
        with c2:
            f_aadhaar = st.text_input("Aadhaar Number", data.get("Aadhaar", ""))
            f_pan = st.text_input("PAN Number", data.get("PAN", ""))
            f_voter = st.text_input("Voter ID", data.get("Voter ID", ""))
            f_dl = st.text_input("Driving License", data.get("Driving License", ""))
        
        # UI Update to reflect ROI extraction
        f_address = st.text_area("Permanent Address (ROI Cleaned)", data.get("Address", ""))

        st.divider()
        st.subheader("🛡️ Security & Export")
        mask_on = st.checkbox("Enable Secure Masking", value=True)
        
        if mask_on:
            st.info(f"💡 Preview Masked Aadhaar: {mask_number(f_aadhaar)}")

        if st.button("💾 Generate Final PDF"):
            final_a = mask_number(f_aadhaar) if mask_on else f_aadhaar
            final_p = mask_number(f_pan) if mask_on else f_pan
            
            pdf_bytes = create_pdf(f_name, final_p, final_a, f_voter, f_dl, f_dob, f_gender, f_address)
            st.download_button("📥 Download Document", pdf_bytes, "application.pdf", "application/pdf")
    else:
        st.info("👋 Upload a document to begin.")
