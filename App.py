import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
import uuid
from docx import Document
from io import BytesIO
from reportlab.pdfgen import canvas
import random
import string
from streamlit_gsheets import GSheetsConnection
import time
import requests
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PneumoniaLens AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# ---------------- GOOGLE SHEETS SETUP ----------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1GJVs3wJwKQzcmDGHICowY0GRn86U6gaFk7I1dNxgbqM/edit?gid=1208527069#gid=1208527069"
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbwbd4bPsxiDVuZtNfd8VbydP3NKdlf2-dl3hXOy-z8bGzZIUPbWFaRNOhacdU7L2MKt/exec"
conn = st.connection("gsheets", type=GSheetsConnection)

def get_doctors_db():
    try:
        return conn.read(spreadsheet=SHEET_URL, worksheet="Doctors", ttl=0)
    except Exception:
        return pd.DataFrame(columns=["ID", "Name", "Email", "Department", "Password"])

def update_doctors_db(df):
    conn.update(spreadsheet=SHEET_URL, worksheet="Doctors", data=df)

def get_logs_db():
    try:
        return conn.read(spreadsheet=SHEET_URL, worksheet="Logs", ttl=0)
    except Exception:
        return pd.DataFrame(columns=["Scan ID", "Operator", "Result", "Confidence", "Timestamp"])

def update_logs_db(df):
    conn.update(spreadsheet=SHEET_URL, worksheet="Logs", data=df)

# ---------------- SESSION STATE INIT ----------------
if "logged_in_doctor" not in st.session_state:
    st.session_state.logged_in_doctor = None
if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""
if "logged_in_admin" not in st.session_state:
    st.session_state.logged_in_admin = False
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()

# ---> NEW: Toast Queue Logic <---
if "show_login_toast" not in st.session_state:
    st.session_state.show_login_toast = False
if "switch_to_scan" not in st.session_state:
    st.session_state.switch_to_scan = False
# If there is a toast queued up, show it instantly and then clear the queue!
if st.session_state.show_login_toast:
    st.toast(st.session_state.show_login_toast, icon="✅")
    st.session_state.show_login_toast = False
# ---------------- SECURITY: AUTO-LOCK (2 MINS) ----------------
def check_inactivity():
    if st.session_state.logged_in_doctor or st.session_state.logged_in_admin:
        now = datetime.now()
        elapsed = (now - st.session_state.last_activity).total_seconds()
        if elapsed > 120:  # 2 Minutes
            st.session_state.logged_in_doctor = None
            st.session_state.doctor_name = ""       # <--- Clear name on timeout
            st.session_state.logged_in_admin = False
            st.session_state.last_activity = now
            st.warning("Session expired due to 2 minutes of inactivity.")
            st.rerun()

def update_activity():
    st.session_state.last_activity = datetime.now()

check_inactivity()

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp { background-color:#000000; color:white; }
.stButton>button { background-color:#00FFFF; color:black; font-weight:bold; border-radius:8px; width: 100%; }
h1,h2,h3 { color:#00FFFF !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #333; padding-bottom: 5px; }
.stTabs [data-baseweb="tab"] { color: white; border-radius: 4px; padding: 10px 20px; background-color: #1a1a1a; transition: 0.3s; }
.stTabs [aria-selected="true"] { background-color: #00FFFF !important; color: black !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL & GRAD-CAM ----------------
@st.cache_resource
def load_model():
    model_path = "pneumonia_3class_97_perfection.h5"
    # The direct link you just found
    model_url = "https://huggingface.co/Ronit-0/PneumoniaLens-Weights/resolve/main/pneumonia_3class_97_perfection.h5?download=true"

    if not os.path.exists(model_path):
        with st.spinner("☁️ AI Model not found. Downloading from Cloud..."):
            try:
                response = requests.get(model_url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error("❌ Cloud connection failed.")
                    return None
            except Exception as e:
                st.error(f"❌ Download Error: {e}")
                return None

    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()
CLASS_NAMES = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRAL']

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="relu"):
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list): preds = tf.convert_to_tensor(preds[0])
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ---------------- TOP HEADER DASHBOARD ----------------
col_head1, col_head2, col_head3 = st.columns([2, 1, 1])
with col_head1:
    st.markdown("<h1 style='margin-bottom: 0px;'>🫁 Pneumonia Lens</h1>", unsafe_allow_html=True)
    st.caption("v2.0 DenseNet121 | 3-class Architecture | True Accuracy - 81.28%")
with col_head2:
    st.write("") 
    if st.session_state.logged_in_admin:
        st.success("Admin Session Active")
    elif st.session_state.logged_in_doctor:
        # <--- NEW: Display the Doctor's Name instead of ID
        st.success(f"👨‍⚕️ Dr. {st.session_state.doctor_name}") 
    else:
        st.info("System: Restricted Access")
with col_head3:
    st.write("") 
    if st.session_state.logged_in_admin or st.session_state.logged_in_doctor:
        if st.button("Secure Logout"):
            st.session_state.logged_in_doctor = None
            st.session_state.doctor_name = ""       # <--- Clear name on logout
            st.session_state.logged_in_admin = False
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- NAVIGATION TABS ----------------
tab_home, tab_login, tab_scan, tab_analytics, tab_doc = st.tabs([
    "🏠 Home",
    "🔐 Login Portal",
    "🔍 AI Diagnostic Scan",
    "📊 Model Analytics",
    "📖 Documentation"
])


# =========================================================
# ======================= HOME ============================
# =========================================================
with tab_home:
    update_activity()
    st.title("Welcome to PneumoniaLens AI")
    st.subheader("Advanced Clinical Second-Opinion System")
    st.markdown("""
    ### 🎯 Our Mission: Bridging the Diagnostic Gap
    In a highly stressful and dynamic environment, the sheer volume of thoracic images can result in "diagnostic fatigue." Our mission is to offer a second pair of eyes that uses AI to specialize in the critical 3-way differential diagnosis between normal lungs and the two different forms of pneumonia.
    
    **Clinically Honest Accuracy:** While other models boast a purported '99%' accuracy rate for a simple binary set, we're more realistic with a grounded accuracy rate of '81.28%' for the much more difficult 3-class problem, which acknowledges the visual overlap between viral and bacterial patterns. This is a "transparency-first" tool for radiologists.
    
    **Rapid Triage:** With the ability to analyze 60 layers of DenseNet121 architecture in mere seconds, this is a digital filter that works before the human eye ever lays a hand on the image.
    
    **Standardizing Care:** Our goal is to eliminate the subjective element of interpretation. Whether the image is the first of the morning or the last of a 12-hour graveyard shift, the AI provides a baseline.
    """)

    st.markdown("""
    ### 🏥 Practical Use Cases & Real-World Handling
    1.  **Emergency Department Triage** - In a crowded ER, patients with respiratory distress need immediate prioritization.
    *The Situation:* Ten X-rays arrive simultaneously.
    *How the Model Handles It:* The AI instantly scans the batch. It identifies a "Bacterial" signature with high confidence in one patient, signaling a potential lobar consolidation that requires immediate antibiotics.
    
    2.  **Viral vs. Bacterial Differentiation** - This is one of the hardest visual tasks in radiology. Viral pneumonia often presents as "patchy" or ground-glass opacities, while bacterial is usually more "consolidated."
    *The Situation:* A patient presents with fever and cough; the visual symptoms are ambiguous.
    *How the Model Handles It:* Our model uses its deep-feature extraction to look for subtle texture differences. By providing a Confidence Metric, it assists the clinician in deciding between antibiotics or antivirals.
    
    3.  **Grad-CAM Localization for Fatigue Reduction** - Human error often occurs because a doctor misses a small area of interest due to exhaustion.
    *The Situation:* A radiologist is on their 100th scan of the day.
    *How the Model Handles It:* The Grad-CAM Focus Map acts as a "Heatmap of Interest." It highlights the specific region of the lung that triggered the AI's classification.
    """)

    st.markdown("<h3 style='text-align:center;'>Visual Diagnostic Targeting</h3>", unsafe_allow_html=True)
# Updated path to include 'assets/'
gradcam_path = "assets/gradcam_final_overlay.png" 

if os.path.exists(gradcam_path):
    st.image(gradcam_path, use_container_width=True)
    st.info('Image generated using XAI logic...')
else:
    st.error(f"⚠️ Image not found at: {gradcam_path}")


# =========================================================
# ================= LOGIN PORTAL ==========================
# =========================================================
with tab_login:
    update_activity()
    st.title("🔐 Authentication & Management")
    
    # The 3-Tab System
    sub_tab_register, sub_tab_login, sub_tab_admin = st.tabs(["📝 Register", "👨‍⚕️ Doctor Login", "🛡️ Admin Access"])

    # --- 1. REGISTRATION TAB ---
    with sub_tab_register:
        st.subheader("Medical Staff Registration")
        st.info("The system will automatically generate a secure 6-digit Doctor ID and Password.")
        
        with st.form("registration_form", clear_on_submit=True):
            reg_first = st.text_input("First Name")
            reg_last = st.text_input("Last Name")
            reg_email = st.text_input("Professional Email")
            reg_dept = st.selectbox("Department", ["Radiology", "Emergency Triage", "Internal Medicine", "General Practice"])

            submit_reg = st.form_submit_button("Generate Credentials & Register")

            if submit_reg:
                if reg_first and reg_last and reg_email:
                    with st.spinner("Connecting to secure Cloud API..."):
                        # Package the data to send to Google
                        payload = {
                            "firstName": reg_first,
                            "lastName": reg_last,
                            "email": reg_email,
                            "dept": reg_dept
                        }
                        
                        try:
                            # Send the data to your Google Apps Script
                            response = requests.post(APPS_SCRIPT_URL, json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                if result.get("status") == "success":
                                    new_id = result["id"]
                                    new_pass = result["password"]
                                    
                                    st.success("✅ Registration Successful! Credentials securely generated by Cloud Backend.")
                                    st.code(f"Doctor ID: {new_id}\nPassword: {new_pass}", language="text")
                                else:
                                    st.error(f"Backend Error: {result.get('message')}")
                            else:
                                st.error("Cloud server error. Could not connect to API.")
                        except Exception as e:
                            st.error(f"Failed to connect to backend API: {e}")
                else:
                    st.error("Please fill in all required fields (Name and Email).")
    # --- 2. DOCTOR LOGIN TAB ---
    with sub_tab_login:
        st.subheader("Doctor Login")
        
        with st.form("doctor_login_form", clear_on_submit=True):
            log_id = st.text_input("Doctor ID (6-digit number)")
            log_pw = st.text_input("Password", type="password")
            submit_doc = st.form_submit_button("Login")

            if submit_doc:
                # 1. Loading animation happens here
                with st.spinner("Verifying with Database..."):
                    st.cache_data.clear()
                    docs_df = get_doctors_db()
                    
                    docs_df["ID"] = docs_df["ID"].astype(str).str.split('.').str[0].str.strip()
                    docs_df["Password"] = docs_df["Password"].astype(str).str.strip()
                    
                    clean_id = str(log_id).strip()
                    clean_pw = str(log_pw).strip()
                    
                    match = docs_df[(docs_df["ID"] == clean_id) & (docs_df["Password"] == clean_pw)]
                
                # 2. TOAST NOTIFICATION AND AUTO-CLICKER QUEUE
                if not match.empty:
                    st.session_state.logged_in_doctor = clean_id
                    doctor_name = match.iloc[0]["Name"]
                    st.session_state.doctor_name = doctor_name 
                    
                    st.session_state.show_login_toast = f"Authenticated successfully: Dr. {doctor_name}"
                    st.session_state.switch_to_scan = True  # <--- FIRE THE AUTO-CLICKER
                    
                    st.rerun()
                else:
                    st.error("Invalid Doctor ID or Password.")
                        
    # --- 3. ADMIN LOGIN TAB ---
    with sub_tab_admin:
        st.subheader("System Administrator Login")
        
        # Pull Admin Credentials from secrets.toml securely
        try:
            ADMIN_ID = st.secrets["admin"]["id"]
            ADMIN_PW = st.secrets["admin"]["password"]
        except KeyError:
            st.error("⚠️ Admin credentials not found in secrets.toml! Defaulting to admin/admin")
            ADMIN_ID = "admin"
            ADMIN_PW = "admin"
        
        # Only show the login form if the admin is NOT logged in
        if not st.session_state.logged_in_admin:
            with st.form("admin_login_form", clear_on_submit=True):
                a_u = st.text_input("Admin ID")
                a_p = st.text_input("Admin Password", type="password")
                submit_admin = st.form_submit_button("Authorize Admin")
                
                if submit_admin:
                    if a_u == ADMIN_ID and a_p == ADMIN_PW:
                        st.session_state.logged_in_admin = True
                        st.rerun()
                    else:
                        st.error("Unauthorized access.")
        else:
            # Admin Control Panel (Visible only when logged in)
            st.success("✅ Global Admin access granted.")

            st.markdown("---")
            st.markdown("### 📋 Administrator Control Panel")
            
            st.markdown("#### ➕ Register Medical Staff")
            with st.expander("Open Registration Form"):
                with st.form("admin_doc_reg_form", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        admin_reg_first = st.text_input("First Name")
                        admin_reg_email = st.text_input("Professional Email")
                    with col2:
                        admin_reg_last = st.text_input("Last Name")
                        admin_reg_dept = st.selectbox("Department", ["Radiology", "Emergency Triage", "Internal Medicine", "General Practice"])
                    
                    submit_admin_reg = st.form_submit_button("Generate Credentials & Register")

                    if submit_admin_reg:
                        if admin_reg_first and admin_reg_last and admin_reg_email:
                            with st.spinner("Sending request to Cloud API..."):
                                # Package the ADMIN's input data to send to Google
                                payload = {
                                    "firstName": admin_reg_first,
                                    "lastName": admin_reg_last,
                                    "email": admin_reg_email,
                                    "dept": admin_reg_dept
                                }
                                
                                try:
                                    # Send the data to your Google Apps Script
                                    response = requests.post(APPS_SCRIPT_URL, json=payload)
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        if result.get("status") == "success":
                                            new_id = result["id"]
                                            new_pass = result["password"]
                                            
                                            st.success(f"✅ Doctor Registered Successfully via API!")
                                            st.code(f"Doctor ID: {new_id}\nPassword: {new_pass}", language="text")
                                        else:
                                            st.error(f"Backend Error: {result.get('message')}")
                                    else:
                                        st.error("Cloud server error. Could not connect to API.")
                                except Exception as e:
                                    st.error(f"Failed to connect: {e}")
                        else:
                            st.error("Please fill in all required fields.")
            
            st.markdown("#### Cloud Registered Doctors")
            docs_df = get_doctors_db()
            if docs_df.empty:
                st.info("No doctors registered yet.")
            else:
                st.dataframe(docs_df.drop(columns=["Password"], errors="ignore"), use_container_width=True, hide_index=True)
                
                # Revoke Access via form to keep it clean
                with st.form("revoke_form", clear_on_submit=True):
                    revoke_id = st.text_input("Enter Doctor ID to Revoke:")
                    submit_revoke = st.form_submit_button("Revoke Access")
                    
                    if submit_revoke:
                        st.cache_data.clear()
                        docs_df = get_doctors_db()
                        
                        # Chop off the ".0" for the revocation check too
                        docs_df["ID"] = docs_df["ID"].astype(str).str.split('.').str[0].str.strip()
                        clean_revoke_id = str(revoke_id).strip()
                        
                        if clean_revoke_id in docs_df["ID"].values:
                            docs_df = docs_df[docs_df["ID"] != clean_revoke_id]
                            update_doctors_db(docs_df)
                            st.success(f"Access revoked for ID: {clean_revoke_id}")
                            st.rerun()
                        else:
                            st.error("Doctor ID not found in database.")
            
            st.markdown("#### Cloud Diagnostic Logs")
            logs_df = get_logs_db()
            if logs_df.empty:
                st.info("No diagnostic history found.")
            else:
                st.dataframe(logs_df, use_container_width=True, hide_index=True)

# =========================================================
# =================== DIAGNOSTIC ==========================
# =========================================================
with tab_scan:
    update_activity()
    st.title("AI Diagnostic Scan")

    # RESTRICTION LOGIC: Check for Doctor specifically
    if not st.session_state.logged_in_doctor:
        if st.session_state.logged_in_admin:
            st.error("🛑 Security Block: Administrators are restricted from performing clinical diagnostic scans. You must be logged in as a licensed Doctor.")
        else:
            st.warning("Please log in through the Portal with a valid Doctor ID to use the scanner.")
    elif model is None:
        st.error("Model 'pneumonia_3class_97_perfection.h5' not found.")
    else:
        confidence_threshold = st.slider("Set Diagnostic Confidence Threshold", 0.0, 1.0, 0.50, 0.01)
        st.caption("Lowering this increases detection speed; raising it reduces 'false positives'.")
        uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

        if uploaded_file:
            with st.status("DenseNet121 scanning through 60 layers, please wait...") as status:
                pil_img = Image.open(uploaded_file).convert("RGB")
                orig_width, orig_height = pil_img.size
                img_res = pil_img.resize((224,224))
                img_arr = np.array(img_res)/255.0
                img_in = np.expand_dims(img_arr, axis=0)

                preds_raw = model.predict(img_in)[0]
                idx = np.argmax(preds_raw)
                confidence = preds_raw[idx] 
                raw_label = CLASS_NAMES[idx]

                if confidence >= confidence_threshold:
                    final_label = raw_label
                    status_color = "success"
                else:
                    final_label = "UNCERTAIN / MANUAL REVIEW REQUIRED"
                    status_color = "error"

                heatmap = make_gradcam_heatmap(img_in, model)
                h_res = cv2.resize(heatmap, (orig_width, orig_height)) 
                h_col = cv2.applyColorMap(np.uint8(255 * h_res), cv2.COLORMAP_JET)
                h_rgb = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
                
                orig_cv2 = np.array(pil_img)
                overlay = cv2.addWeighted(orig_cv2, 0.6, h_rgb, 0.4, 0)
                
                status.update(label="Scanned successfully!", state="complete", expanded=False)

            # SAVE LOGS TO GOOGLE SHEETS
            logs_df = get_logs_db()
            new_log = pd.DataFrame([{
                "Scan ID": str(uuid.uuid4())[:8],
                "Operator": st.session_state.logged_in_doctor,
                "Result": final_label,
                "Confidence": f"{confidence*100:.2f}%",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            logs_df = pd.concat([logs_df, new_log], ignore_index=True)
            update_logs_db(logs_df)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption=f"Original ({orig_width}x{orig_height})", use_container_width=True)
            with col2:
                st.image(overlay, caption=f"Grad-CAM Heatmap ({orig_width}x{orig_height})", use_container_width=True)

            st.metric("Diagnosis", final_label)
            st.metric("Confidence", f"{confidence*100:.2f}%")

            def generate_pdf():
                buffer = BytesIO()
                c = canvas.Canvas(buffer)
                c.drawString(50,800,f"PneumoniaLens AI Report")
                c.drawString(50,780,f"Result: {final_label}")
                c.drawString(50,760,f"Confidence: {confidence*100:.2f}%")
                c.save()
                buffer.seek(0)
                return buffer

            st.download_button("Download PDF Report", generate_pdf(), file_name=f"report_{final_label}.pdf")


# =========================================================
# =================== ANALYTICS ===========================
# =========================================================
with tab_analytics:
    update_activity()
    st.title("Model Analytics")
    st.markdown("""
    ### 🔧 Development methodology
    * **Base Architecture:** DenseNet121 (Pre-trained on ImageNet).
    * **Fine-Tuning:** 60 layers unfrozen to capture specific thoracic cavity features.
    * **Class Balancing:** Addressed extreme overlap between Viral and Bacterial presentations using strict `class_weight_dict`.
    """)
    st.markdown("""
    ### 🧮 Hardware Performance Metrics
    Developed locally and trained on cloud via T4 GPU from google colab. 
    Trained through 30 epochs to give highest possible accuracy without overfitting. 
    The final model achieved a clinically honest '81.28%' accuracy on the 3-class problem, which is a significant improvement over the binary classification baseline and reflects the real-world difficulty of differentiating between viral and bacterial pneumonia on chest X-rays.
    """)

    st.markdown("<h3 style='text-align:center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
# Updated path to include 'assets/'
matrix_path = "assets/final_confusion_matrix_3class.png"

if os.path.exists(matrix_path):
    st.image(matrix_path, use_container_width=True)
else:
    st.error(f"⚠️ Image not found at: {matrix_path}")
    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>Cloud Diagnostic Logs</h3>", unsafe_allow_html=True)

    # Fetch logs from Google Sheets
    logs_df = get_logs_db()
    if logs_df.empty:
        st.info("No cloud diagnostic history found.")
    else:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)


# =========================================================
# ================= DOCUMENTATION =========================
# =========================================================
with tab_doc:
    update_activity()
    st.title("📃Documentation")
    st.markdown("""
    ### DenseNet121 – Clinical Architecture
    DenseNet-121 is a 121-layer Convolutional Neural Network (CNN) architecture. DenseNet-121 enhances the efficiency of Deep Learning by providing a direct connection between each layer and every other layer in a feed-forward manner. DenseNet-121 uses "dense blocks" to concatenate feature maps, which reduces the number of parameters.
    
    ### Key features
    * Dense Connectivity: Unlike other CNNs, DenseNet-121 has every layer connected to all subsequent layers in a block. This encourages feature reuse and solves the vanishing gradient problem.
    * Structure: It has an initial 7x7 convolutional layer. After that, four dense blocks are implemented. Each block has 6, 12, 24, and 16 layers respectively. Transition layers are implemented between them.
    * Efficiency: It has a 32-filter growth rate. This makes the network computationally efficient and improves feature usage compared to ResNet.
    * Input Size: Images are inputted into the network.
    """)

    st.title("❔Most Probable FAQs")
    st.markdown("<h4 style='color:yellow;'>1. Why DenseNet121 over ResNet or VGG16?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>DenseNet121 utilizes dense connections to maximize feature reuse and gradient flow, allowing the model to capture subtle medical patterns with significantly fewer parameters than VGG or ResNet.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>2. How was the overfitting handled?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>We combined data augmentation (rotation, zooming) with dropout layers to prevent the model from memorizing the training set, ensuring it generalizes to real-world clinical images.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>3. What is the clinical value of Grad-CAM?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>Grad-CAM provides Explainable AI (XAI) by overlaying heatmaps on X-rays; this allows radiologists to verify that the AI is focusing on actual pathology rather than image noise.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>4. Why use an adjustable classification threshold?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>It allows clinicians to tune the model for high sensitivity in emergency triage (missing no cases) or high specificity in routine screening (reducing false alarms).</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>5. How is a '0.0%' idle CPU load maintained?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>By using @st.cache_resource, the 31MB model is loaded into RAM once upon startup; this eliminates redundant processing and keeps the app responsive without taxing the CPU during idle periods.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:yellow;'>6. How was the '81.28%' accuracy validated?</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:green;'>The metric was established on a rigorous, unseen test split and further verified through 'in-the-wild' testing with diverse external images to ensure robust real-world performance.</p>", unsafe_allow_html=True)

# =========================================================
# ================= JAVASCRIPT AUTO-CLICKER ===============
# =========================================================
if st.session_state.switch_to_scan:
    # Immediately turn it off so it doesn't keep clicking every time you do something
    st.session_state.switch_to_scan = False 
    
    # Inject the hidden JavaScript into the browser
    components.html(
        """
        <script>
        // Streamlit puts custom code in a protective box (iframe). 
        // We use window.parent.document to reach out of the box to the main page.
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        
        // Loop through all the tabs until we find the Diagnostic Scan one, then click it!
        for (let i = 0; i < tabs.length; i++) {
            if (tabs[i].innerText.includes("AI Diagnostic Scan")) {
                tabs[i].click();
                break;
            }
        }
        </script>
        """,
        height=0, width=0

    )

