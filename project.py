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
from streamlit_cropper import st_cropper
import tempfile
import os 

# --- NEW: AUTOMATIC DOCUMENT EDGE DETECTION & CROP ---
def crop_to_document(image):
    orig = image.copy()
    h_orig, w_orig = image.shape[:2]
    
    # 1. Standardize size for detection
    ratio = h_orig / 500.0
    image_resized = cv2.resize(image, (int(w_orig / ratio), 500))
    
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Use a slightly less aggressive Canny to find the outer border
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area so we check the biggest shapes first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            # Calculate Area
            area = cv2.contourArea(c)
            # Ignore shapes that are too small (like QR codes which are usually < 10% of total area)
            if area < (image_resized.shape[0] * image_resized.shape[1] * 0.2):
                continue
                
            pts = approx.reshape(4, 2) * ratio
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-Left
            rect[2] = pts[np.argmax(s)] # Bottom-Right

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-Right
            rect[3] = pts[np.argmax(diff)] # Bottom-Left

            (tl, tr, br, bl) = rect
            
            # --- Aspect Ratio Check ---
            width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            
            # Aadhaar cards are horizontal. If detected shape is a tall vertical 
            # or a tiny square, skip it.
            if width < height: 
                continue

            maxWidth, maxHeight = int(width), int(height)
            dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
            return warped

    # Fallback: if no large card is found, return original (don't crop to a random QR code)
    return orig 

# --- STEP 1: UPDATED EXTRACTION LOGIC ---
def extract_fields(text, img):
    # Standard clean up
    clean_text = re.sub(r'[:|]', '', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # --- 1. ID Number Extraction ---
    all_numbers = re.findall(r'\b\d{4}\s\d{4}\s\d{4}(?:\s\d{4})?\b|\b\d{12,16}\b', clean_text)
    aadhaar = next((num for num in all_numbers if len(num.replace(" ", "")) == 12), "")

    pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text.upper())
    voter = re.search(r'[A-Z]{3}[0-9]{7}', text.upper())
    dl = re.search(r'[A-Z]{2}[0-9]{2}\s?[0-9]{11}', text.upper())
    dob = re.search(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', text)
    
    gender = "Male"
    if re.search(r'FEMALE|महिला|F\s?E\s?M\s?A\s?L\s?E', text, re.IGNORECASE):
        gender = "Female"

    # --- 2. RESILIENT ADDRESS EXTRACTION WITH STRICT CLEANING ---
    address = ""
    address_raw = ""
    
    # Keywords that mark the end of an address block
    stop_keywords = [
        r'Mobile', r'Phone', r'दूरभाष', r'Help', r'www', r'unique', 
        r'VID', r'Aadhaar', r'आच्चार', r'HATH', r'Your', r'No\.', r'पत्ता', r'Photograph'
    ]
    pattern = '|'.join(stop_keywords)
    addr_starts = ['S/O', 'D/O', 'W/O', 'C/O', 'ADDRESS', 'पता', 'पता:']

    # Use pytesseract data for Geometric ROI
    data = pytesseract.image_to_data(img, lang='eng+hin+mar', output_type=pytesseract.Output.DICT)
    start_x, start_y = -1, -1
    
    for i, word in enumerate(data['text']):
        if any(marker in word.upper() for marker in addr_starts):
            start_x, start_y = data['left'][i], data['top'][i]
            break
            
    if start_x != -1:
        h, w, _ = img.shape
        roi_x, roi_y = max(0, start_x - 10), max(0, start_y - 10)
        roi_w, roi_h = int(w), int(h * 0.4) 
        
        crop_address_img = img[roi_y:min(roi_y+roi_h, h), roi_x:min(roi_x+roi_w, w)]
        address_raw = pytesseract.image_to_string(crop_address_img, lang='eng+hin+mar').strip()

    # FALLBACK: If ROI extraction yielded nothing, scan the full text
    if not address_raw or len(address_raw) < 10:
        fallback_match = re.search(r'(?:Address|पता)[:\s]+([\s\S]+?)(?=Mobile|Phone|Help|www|unique|$)', text, re.IGNORECASE)
        if fallback_match:
            address_raw = fallback_match.group(1).strip()

    # --- STRICT CLEANING LOGIC (ENHANCED) ---
    if address_raw:
        # 1. Truncate at first Stop Word (removes Phone, VID, footer junk)
        address_clean = re.split(pattern, address_raw, flags=re.IGNORECASE)[0]
        
        # 2. NEW: Truncate at any 10-digit number (usually the mobile number)
        address_clean = re.split(r'\d{10}', address_clean)[0]

        # 3. NEW: Truncate at the word 'QR' or 'Photograph' 
        # (common in misread Aadhaar footers)
        address_clean = re.split(r'QR|Code|Photograph|Photo|vid', address_clean, flags=re.IGNORECASE)[0]
        
        # 4. Remove Father's Name if it exists at the start
        address_clean = re.sub(r'^(S/O|D/O|W/O|C/O)[:\s]+[^,\n]+[,\n]', '', address_clean, flags=re.IGNORECASE).strip()
        
        # 5. Remove leftover labels and OCR artifacts
        for m in addr_starts:
            address_clean = re.sub(m, '', address_clean, flags=re.IGNORECASE)
            
        address_clean = re.sub(r'[|=|_|—|–|#|*]', '', address_clean)
        
        # 6. Final Polish: remove trailing commas or dots left after splitting
        address = re.sub(r'\s+', ' ', address_clean).strip()
        address = address.strip(' ,.-')

    # --- 3. Name Extraction Logic ---
    garbage_keywords = ['GOVERNMENT', 'INDIA', 'INCOME', 'TAX', 'DEPARTMENT', 'CARD', 'NUMBER', 'SIGNATURE', 'UNIQUE', 'LICENSE', 'VOTER', 'EPIC', 'VID']
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
# --- STEP 2: FACE DETECTION ---
def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60))
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        face_img = image[max(0, y-pad_h):min(image.shape[0], y+h+pad_h), max(0, x-pad_w):min(image.shape[1], x+w+pad_w)]
        return face_img
    return None

def mask_number(number):
    if not number or len(str(number)) < 4: return number
    return "XXXX-XXXX-" + str(number)[-4:]

# --- STEP 3: PDF GENERATION (WITH FACE) ---
def create_pdf(name, pan, aadhaar, voter, dl, dob, gender, address, face_img=None):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header Section
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "VERIFIED CITIZEN APPLICATION", 1, 1, 'C', True)
    pdf.ln(10)

    # 2. Store Table Start Position
    table_start_y = pdf.get_y()
    
    # Data for the table
    data = [
        ["Full Name", name], 
        ["DOB", dob], 
        ["Gender", gender], 
        ["Aadhaar", aadhaar], 
        ["PAN", pan], 
        ["Voter ID", voter], 
        ["DL", dl]
    ]

    # 3. Render Table Rows with Photo Pocket
    for i, row in enumerate(data):
        row_value_width = 100 if i < 4 else 0 
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, row[0], 1, 0, 'L')
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(row_value_width, 10, str(row[1]) if row[1] else "N/A", 1, 1, 'L')

    # 4. Place Photo Aligned with Name/DOB/Gender Rows
    if face_img is not None:
        try:
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img_pil.save(tmp.name, format="JPEG")
                temp_path = tmp.name
            
            photo_x, photo_y, photo_w, photo_h = 150, table_start_y, 40, 40
            
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.5)
            pdf.rect(photo_x, photo_y, photo_w, photo_h)
            
            pdf.image(temp_path, x=photo_x + 1, y=photo_y + 1, w=photo_w - 2, h=photo_h - 2)
            os.remove(temp_path)
        except Exception as e:
            st.warning(f"Could not add face to PDF: {e}")

    # 5. Systematic Address Section
    pdf.set_font("Arial", 'B', 12)
    address_text = str(address) if address else "N/A"
    
    pdf.cell(40, 20, "Address", 1, 0, 'L') 
    
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 10, address_text, 1, 'L')
    
    return pdf.output(dest='S').encode('latin-1')

# --- STEP 4: STREAMLIT UI ---
st.set_page_config(page_title="Citizen Digital Assistant", layout="wide")

if "final_data" not in st.session_state: st.session_state.final_data = None
if "detected_face" not in st.session_state: st.session_state.detected_face = None
if "otp_sent" not in st.session_state: st.session_state.otp_sent = False
if "rotation_angle" not in st.session_state: st.session_state.rotation_angle = 0

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
                st.session_state.final_data = {"Name": "Yash Mahajan", "DOB": "01/01/1990", "Gender": "Male", "Aadhaar": "1234 5678 9012", "PAN": "ABCDE1234F", "Address": "House No 123, City, State"}
            else: st.error("Invalid OTP")

st.title("🇮🇳 AI Form Filling Assistant")
tab1, tab2 = st.tabs(["📤 Upload ID", "📄 Final Application"])

with tab1:
    st.markdown("### 📥 Document Upload")
    st.info("💡 **Supported Documents**: You can upload or scan your **Aadhaar Card, PAN Card, Voter ID, or Driving License**. The AI will automatically detect the document type and extract your details.")
    
    source = st.radio("Select Source:", ["Upload File", "Take Photo"])
    uploaded_file = st.camera_input("Scan Document") if source == "Take Photo" else st.file_uploader("Upload ID", type=["jpg", "png", "jpeg", "pdf"])
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        
        if uploaded_file.name.endswith(".pdf"):
            images = convert_from_bytes(file_bytes)
            img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.warning("⚠️ **Note:** Please ensure image is horizontal for best OCR accuracy.")
        
        col_rot1, col_rot2, col_rot3 = st.columns([1, 1, 4])
        with col_rot1:
            if st.button("🔄 Rotate Left"):
                st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
        with col_rot2:
            if st.button("🔄 Rotate Right"):
                st.session_state.rotation_angle = (st.session_state.rotation_angle - 90) % 360
        
        if st.session_state.rotation_angle % 360 != 0:
            angle = st.session_state.rotation_angle
            if angle == 90 or angle == -270: img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180 or angle == -180: img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)
            elif angle == 270 or angle == -90: img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)

        st.divider()
        crop_mode = st.checkbox("🛠️ Manual Adjust Crop", value=False)

        if crop_mode:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            cropped_pil = st_cropper(img_pil, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
        else:
            with st.status("Detecting Document Edges..."):
                img = crop_to_document(img_cv)
                st.write("✅ Document Rectified.")

        col_img, col_face = st.columns([2, 1])
        with col_img: 
            st.image(img, caption="Document Preview", use_container_width=True)
        
        with col_face:
            face = extract_face(img)
            if face is not None:
                st.session_state.detected_face = face
                st.image(face, caption="Detected Face", width=150)
            else: st.info("No face detected.")

        if st.button("🚀 Run AI Extraction"):
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🎨 Pre-processing...")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                processed_img = cv2.medianBlur(gray, 3)
                progress_bar.progress(40)
                
                status_text.text("🔍 Scanning (Eng+Hin+Mar)...")
                raw_text = pytesseract.image_to_string(processed_img, lang='eng+hin+mar')
                progress_bar.progress(70)
                
                status_text.text("📂 Mapping Fields...")
                extracted_data = extract_fields(raw_text, img)
                st.session_state.final_data = extracted_data
                
                progress_bar.progress(100)
                latency = round(time.time() - start_time, 2)
                status_text.text(f"✅ Complete! Latency: {latency}s")
                
                if latency <= 5: st.success(f"Performance Target Met: {latency}s")
                else: st.warning(f"Latency ({latency}s) exceeded 5s target")
                
                st.balloons()
            except Exception as e:
                st.error(f"Extraction failed: {e}")

with tab2:
    if st.session_state.final_data:
        data = st.session_state.final_data
        st.subheader("📝 Edit & Verify Details")
        
        # --- IMPROVED VOICE INPUT SECTION ---
        st.info("🎤 **Voice Correction**: Record your correction (e.g., 'Change name to Rajesh')")
        audio = mic_recorder(start_prompt="Record Voice", stop_prompt="Stop Recording", key='voice_recorder')

        if audio:
            # Process audio only if new bytes are detected
            r = sr.Recognizer()
            audio_data = io.BytesIO(audio['bytes'])
            with sr.AudioFile(audio_data) as voice_src:
                recorded_audio = r.record(voice_src)
                try:
                    # High-level guideline: Support multiple Indian languages
                    voice_text = r.recognize_google(recorded_audio, language='en-IN')
                    st.success(f"Recognized: {voice_text}")
                    
                    # PERFORMANCE TRACKING / Field Suggestion Logic
                    if "name" in voice_text.lower():
                        st.info("💡 Tip: Use this to update the 'Full Name' field below.")
                except Exception as e:
                    st.error("Speech recognition failed. Please ensure your microphone is enabled in the browser.")
        
        st.divider()
        
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
        
        f_address = st.text_area("Permanent Address", data.get("Address", ""))
        mask_on = st.checkbox("Enable Secure Masking", value=True)

        if st.button("💾 Generate Final PDF"):
            final_a = mask_number(f_aadhaar) if mask_on else f_aadhaar
            final_p = mask_number(f_pan) if mask_on else f_pan
            pdf_bytes = create_pdf(f_name, final_p, final_a, f_voter, f_dl, f_dob, f_gender, f_address, face_img=st.session_state.detected_face)
            st.download_button("📥 Download Application", pdf_bytes, "application.pdf", "application/pdf")
    else:
        st.info("👋 Upload a document to begin.")
