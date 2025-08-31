import streamlit as st
import cv2
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import tempfile
from datetime import datetime
import time

st.set_page_config(
    page_title="VFIT - AI Virtual Try-On",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .step-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .category-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

@st.cache_resource
def load_cascade_classifier():
    try:
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            './haarcascade_frontalface_default.xml'
        ]
        
        face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    st.success(f"Loaded cascade from: {path}")
                    return face_cascade
        
        st.warning("Haar cascade file not found. Using alternative detection method.")
        return None
    except Exception as e:
        st.error(f"Error loading cascade classifier: {str(e)}")
        return None

def process_frame(frame, item_path): 
    face_cascade = load_cascade_classifier()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    else:
        height, width = gray.shape
        faces = [(int(width/4), int(height/4), int(width/2), int(height/3))]
    
    if len(faces) == 0:
        return None
    
    if not os.path.exists(item_path):
        st.error(f"Item image not found: {item_path}")
        return frame
    
    item_img = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
    if item_img is None:
        st.error(f"Failed to load item image: {item_path}")
        return frame
    
    result = frame.copy()
    for (x, y, w, h) in faces:
        if "necklace" in item_path.lower():
            item_height = int(h * 0.5)
            item_width = int(w * 1.5)
            y_offset = int(y + h * 0.8)
            x_offset = int(x - w * 0.25)

            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset, y_offset)

        elif "earring" in item_path.lower():
            item_height = int(h * 0.3)
            item_width = int(w * 0.3)

       
            y_offset_left = int(y + h * 0.3)
            x_offset_left = int(x - w * 0.3)
          
            y_offset_right = int(y + h * 0.3)
            x_offset_right = int(x + w * 0.9)
            
            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset_left, y_offset_left)
            
           
            item_flipped = cv2.flip(item_resized, 1)
            result = overlay_image(result, item_flipped, x_offset_right, y_offset_right)

            
            continue

        elif "glasses" in item_path.lower() or "goggle" in item_path.lower() or "sunglass" in item_path.lower():
           
            gh, gw = item_img.shape[:2]
            scale = w / gw
            item_resized = cv2.resize(item_img, (int(gw * scale), int(gh * scale)))

            rw, rh = item_resized.shape[1], item_resized.shape[0]
            x_offset = x + (w // 2) - (rw // 2)
            y_offset = y + int(h * 0.4) - (rh // 2)

            result = overlay_image(result, item_resized, x_offset, y_offset)

        elif "tiara" in item_path.lower() or "crown" in item_path.lower():
            
            item_height = int(h * 0.4)
            item_width = int(w * 1.2)
            y_offset = int(y - h * 0.3)
            x_offset = int(x - w * 0.1)

            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset, y_offset)

        elif "top" in item_path.lower() or "shirt" in item_path.lower():
            
            item_width = int(w * 2.5)
            item_height = int(h * 3.0)
            x_offset = int(x - w * 0.7)
            y_offset = int(y + h * 1.0)  

            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset, y_offset)

        else:
          
            item_height = int(h * 0.8)
            item_width = int(w * 0.8)
            y_offset = y
            x_offset = x

            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset, y_offset)
    
    return result

def overlay_image(background, overlay, x, y):
    bg_height, bg_width = background.shape[:2]
    
    x = max(0, min(x, bg_width - 1))
    y = max(0, min(y, bg_height - 1))
    
    h, w = overlay.shape[:2]
    
    
    roi_x1 = x
    roi_y1 = y
    roi_x2 = min(x + w, bg_width)
    roi_y2 = min(y + h, bg_height)
    
    if roi_x1 >= bg_width or roi_y1 >= bg_height or roi_x2 <= 0 or roi_y2 <= 0:
        return background
    
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = min(w, bg_width - x)
    overlay_y2 = min(h, bg_height - y)
    
    roi = background[roi_y1:roi_y2, roi_x1:roi_x2]
    
    overlay_portion = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    
    if overlay_portion.shape[2] == 4:  
        overlay_colors = overlay_portion[:, :, :3]
        alpha_mask = overlay_portion[:, :, 3:] / 255.0
        
        roi = (overlay_colors * alpha_mask + roi * (1 - alpha_mask)).astype(np.uint8)
    else:
        roi = overlay_portion
    
    background[roi_y1:roi_y2, roi_x1:roi_x2] = roi
    
    return background

st.sidebar.title(" VFIT")
page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Home", "👔 Wardrobe", "📸 Try-On", "ℹ️ About"]
)

if page == "🏠 Home":
    st.session_state.page = 'home'
elif page == "👔 Wardrobe":
    st.session_state.page = 'wardrobe'
elif page == "📸 Try-On":
    st.session_state.page = 'tryon'
elif page == "ℹ️ About":
    st.session_state.page = 'about'

import base64

def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return None
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

wardrobe_file = "static/images/wordrobe.png"
profile_file = "static/images/personbg.png"
tryon_file   = "static/images/tryon.png"

wardrobe_bg = get_base64_of_bin_file(wardrobe_file)
profile_bg  = get_base64_of_bin_file(profile_file)
tryon_bg    = get_base64_of_bin_file(tryon_file)

CARD_STYLE = """
    text-align:center; 
    padding:15px; 
    border:1px solid #ddd; 
    border-radius:12px; 
    width:220px; 
    height:180px; 
    margin:auto; 
    display:flex; 
    flex-direction:column; 
    justify-content:center;
    box-shadow:0px 4px 6px rgba(0,0,0,0.1);
"""

if st.session_state.page == 'home':
    st.markdown('<h1 class="main-header">Welcome to VFIT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI-powered virtual try-on experience. Upload wardrobe items and profile photos, then create stunning virtual try-ons with the power of AI.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Wardrobe Card
    with col1:
        style = CARD_STYLE
        if wardrobe_bg:
            style += f"background-image:url('data:image/png;base64,{wardrobe_bg}');background-size:cover;background-position:center;"
        st.markdown(
            f"""
            <div class="stat-card" style="{style}"></div>
            <p style="text-align:center; font-weight:bold; margin-top:8px;">Upload Wardrobe Items</p>
            """,
            unsafe_allow_html=True
        )

    with col2:
        style = CARD_STYLE
        if profile_bg:
            style += f"background-image:url('data:image/png;base64,{profile_bg}');background-size:cover;background-position:center;"
        st.markdown(
            f"""
            <div class="stat-card" style="{style}"></div>
            <p style="text-align:center; font-weight:bold; margin-top:8px;">Take Photo</p>
            """,
            unsafe_allow_html=True
        )

    with col3:
        style = CARD_STYLE
        if tryon_bg:
            style += f"background-image:url('data:image/png;base64,{tryon_bg}');background-size:cover;background-position:center;"
        st.markdown(
            f"""
            <div class="stat-card" style="{style}"></div>
            <p style="text-align:center; font-weight:bold; margin-top:8px;">Virtual Try-Ons</p>
            """,
            unsafe_allow_html=True
        )


    
    st.markdown("## Getting Started")
    
    st.markdown("""
    <div class="step-card">
        <h3>1️⃣ Upload Wardrobe Items</h3>
        <p>Take photos of clothing items or accessories and upload them to your virtual wardrobe. Our AI will extract them onto a clean background.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <h3>2️⃣ Add Profile Photos</h3>
        <p>Upload photos of yourself in different poses. These will be used as the base for your virtual try-ons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <h3>3️⃣ Create Virtual Try-Ons</h3>
        <p>Select a profile photo and wardrobe item, then let the AI create a realistic composition of you wearing the item.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Get Started")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Start with Try-On", use_container_width=True, key="home_tryon_btn"):
            st.session_state.page = "tryon"
            st.rerun()

    with col2:
        if st.button("👔 Browse Wardrobe", use_container_width=True, key="home_wardrobe_btn"):
            st.session_state.page = "wardrobe"
            st.rerun()

elif st.session_state.page == 'wardrobe':
    st.markdown('<h1 class="main-header">Virtual Wardrobe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Browse and select from our collection of virtual clothing items</p>', unsafe_allow_html=True)
    
    
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    uploaded_items_file = 'static/uploads/uploaded_items.json'
    if not os.path.exists(uploaded_items_file):
        with open(uploaded_items_file, 'w') as f:
            json.dump([], f)
    
   
    try:
        with open(uploaded_items_file, 'r') as f:
            uploaded_items = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        uploaded_items = []
    
    
    with st.sidebar.expander("📤 Upload New Item"):
        with st.form("upload_form"):
            st.write("### Add New Item to Wardrobe")
            item_name = st.text_input("Item Name", "")
            item_category = st.selectbox(
                "Category",
                ["Necklace", "Earring", "Tiara", "Hat", "Goggle", "T-shirt"]
            )
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
            
            if st.form_submit_button("Upload Item"):
                if uploaded_file is not None and item_name.strip() != "":
                 
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{item_category.lower()}_{timestamp}_{uploaded_file.name}"
                    filepath = os.path.join('static', 'uploads', filename)
                    
                    
                    with open(filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    
                    new_item = {
                        'name': item_name,
                        'category': item_category,
                        'path': filepath,
                        'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    uploaded_items.append(new_item)
                    
                    
                    with open(uploaded_items_file, 'w') as f:
                        json.dump(uploaded_items, f)
                    
                    st.success(f"Successfully uploaded {item_name} to your wardrobe!")
                else:
                    st.warning("Please provide both an item name and select an image file.")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Jewellery", "Earrings", "Hats & Tiaras", "T-shirts", "Glasses", "My Uploads"])
    
    with tab1:
        st.markdown('<h2 class="category-header">Necklaces</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        necklaces = [
            ("Silver Necklace", "static/images/Necklace11.png"),
            ("Gold Necklace", "static/images/Necklace12.png"),
            ("Thread Necklace", "static/images/Necklace15.png"),
            ("Gold Chain", "static/images/Necklace16.png"),
            ("Gold Chain", "static/images/Necklace17.png"),
        ]
        
        for i, (name, path) in enumerate(necklaces):
            with cols[i % 3]:
                
                if not os.path.exists(path):
                    
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 255, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"necklace_try_{name}_{i}"):
                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab2:
        st.markdown('<h2 class="category-header">Earrings</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        earrings = [
            ("Goldwork Earrings", "static/images/Earrings21.png"),
            ("Studs", "static/images/Earrings22.png"),
            ("Studs", "static/images/Earrings23.png"),
            ("Silver Work Earrings", "static/images/Earrings24.png"),
            ("Goldwork Earrings", "static/images/Earrings25.png"),
            ("Studs", "static/images/Earrings26.png"),
        ]
        
        for i, (name, path) in enumerate(earrings):
            with cols[i % 3]:
                
                if not os.path.exists(path):
                    
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (255, 0, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"earring_try_{name}_{i}"):
                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab3:
        st.markdown('<h2 class="category-header">Hats & Tiaras</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        tiaras = [
            ("Tiara", "static/images/Tiaras31.png"),
            ("Tiara", "static/images/Tiaras32.png"),
            ("Tiara", "static/images/Tiaras33.png"),
            ("Tiara", "static/images/Tiaras34.png"),
            ("Tiara", "static/images/Tiaras35.png"),
            ("Tiara", "static/images/Tiaras36.png"),
        ]
        
        for i, (name, path) in enumerate(tiaras):
            with cols[i % 3]:
                
                if not os.path.exists(path):
                   
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (255, 255, 0), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"tiara_try_{name}_{i}"):
                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab4:
        st.markdown('<h2 class="category-header">T-shirts</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        tshirts = [
            ("Orange T-Shirt", "static/images/Tops41.png"),
            ("Pink T-Shirt", "static/images/Tops42.png"),
            ("Orange T-Shirt", "static/images/Tops45.png"),
            ("White T-Shirt", "static/images/Tops43.png"),
            ("Black T-Shirt", "static/images/Tops48.png"),
            ("White and Black Full Sleeves", "static/images/Tops49.png"),
        ]
        
       
        for item in uploaded_items:
            if item['category'].lower() == 't-shirt':
                tshirts.append((item['name'], item['path']))
                
        for i, (name, path) in enumerate(tshirts):
            with cols[i % 3]:
                
                if not os.path.exists(path):
                    
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 0, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"tshirt_try_{name}_{i}"):
                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab5:
        st.markdown('<h2 class="category-header">Glasses</h2>', unsafe_allow_html=True)
        cols = st.columns(2)
        goggles = [
            ("Goggles", "static/images/Sunglasses61.png"),
            ("Sun Glasses", "static/images/Sunglasses62.png"),
            ("Spectacles", "static/images/Sunglasses63.png"),
            ("Sun Glasses", "static/images/Sunglasses64.png"),
            ("Shades", "static/images/Sunglasses65.png"),
            ("Shades", "static/images/Sunglasses66.png"),
        ]
        
        for i, (name, path) in enumerate(goggles):
            with cols[i % 2]:
                
                if not os.path.exists(path):
                    
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 0, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"goggle_try_{name}_{i}_{st.session_state.page}"):

                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab6:
        st.markdown('<h2 class="category-header">My Uploaded Items</h2>', unsafe_allow_html=True)
        
        if not uploaded_items:
            st.info("You haven't uploaded any items yet. Use the upload form in the sidebar to add items to your wardrobe!")
        else:
            cols = st.columns(3)
            for i, item in enumerate(uploaded_items):
                with cols[i % 3]:
                    try:
                        if os.path.exists(item['path']):
                            st.image(item['path'], use_container_width=True)
                            st.write(f"**{item['name']}**")
                            st.caption(f"Category: {item['category']}")
                            st.caption(f"Added: {item['date_added']}")

                           
                            if st.button(f"Try on {item['name']}", key=f"uploaded_try_{item['category']}_{i}_{item['name'].replace(' ', '_')}"):
                                st.session_state.selected_item = item['path']
                                st.session_state.page = 'tryon'
                                st.rerun()

                            
                            if st.button(f"🗑️ Delete {item['name']}", key=f"delete_{i}_{item['name'].replace(' ', '_')}"):
                                file_path = item['path']

                                
                                uploaded_items.pop(i)

                                
                                with open(uploaded_items_file, 'w') as f:
                                    json.dump(uploaded_items, f)

                                
                                if os.path.exists(file_path):
                                    os.remove(file_path)

                                st.success(f"🗑️ Deleted {item['name']} successfully!")
                                st.rerun()
                        else:
                            st.warning(f"Could not find image: {item['path']}")
                    except Exception as e:
                        st.error(f"Error loading {item.get('name', 'item')}: {str(e)}")

        



elif st.session_state.page == 'tryon':
    st.markdown('<h1 class="main-header">Virtual Try-On</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Use your webcam to try on virtual items in real-time</p>', unsafe_allow_html=True)

    
    st.subheader("📷 Camera Feed")
    frame = None
    try:
        camera_input = st.camera_input("Take a photo to try on items")
        if camera_input is not None:
            image = Image.open(camera_input)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception:
        st.error(" Error accessing webcam.")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    
    base_categories = {
        "T-shirts": [
            ("Orange T-Shirt", "static/images/Tops41.png"),
            ("Pink T-Shirt", "static/images/Tops42.png"),
            ("White T-Shirt", "static/images/Tops43.png")
        ],
        "Necklaces": [
            ("Silver Necklace", "static/images/Necklace11.png"),
            ("Gold Necklace", "static/images/Necklace12.png"),
            ("Thread Necklace", "static/images/Necklace15.png"),
        ],
        "Earrings": [
            ("Goldwork Earrings", "static/images/Earrings21.png"),
            ("Studs", "static/images/Earrings22.png"),
            ("Silver Work Earrings", "static/images/Earrings24.png"),
        ],
        "Tiaras": [
            ("Tiara1", "static/images/Tiaras31.png"),
            ("Tiara2", "static/images/Tiaras32.png"),
            ("Tiara3", "static/images/Tiaras33.png"),
        ],
        "Goggles": [
            ("Goggles", "static/images/Sunglasses61.png"),
            ("Sun Glasses", "static/images/Sunglasses62.png"),
            ("Spectacles", "static/images/Sunglasses63.png"),
        ],
        "Hats": []  
    }

    uploaded_items_file = "uploaded_items.json"
    if os.path.exists(uploaded_items_file):
        with open(uploaded_items_file, "r") as f:
            uploaded_items = json.load(f)
    else:
        uploaded_items = []

    category_map = {
        "T-shirt": "T-shirts",
        "Necklace": "Necklaces",
        "Earring": "Earrings",
        "Tiara": "Tiaras",
        "Goggle": "Goggles",
        "Hat": "Hats"
    }

    item_categories = {cat: items.copy() for cat, items in base_categories.items()}
    for item in uploaded_items:
        mapped_cat = category_map.get(item["category"], None)
        if mapped_cat:
            item_categories[mapped_cat].append((item["name"], item["path"]))

   
    if frame is not None:
        selected_item = st.session_state.get('selected_item', 'static/images/Necklace11.png')
        try:
            processed_frame = process_frame(frame, selected_item)
            if processed_frame is not None:
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_frame_rgb, caption="Virtual Try-On Result", use_container_width=True)
                st.session_state.processed_image = processed_frame_rgb
            else:
                st.image(frame, caption="Original Photo", use_container_width=True)
                st.warning("No face detected.")
        except Exception as e:
            st.error(f" Error processing image: {str(e)}")
            st.image(frame, caption="Original Photo", use_container_width=True)

    
    st.subheader("👔 Select Item")
    col_left, col_right = st.columns(2)

    with col_left:
        for category in ["T-shirts", "Necklaces", "Earrings"]:
            if category in item_categories:
                st.write(f"**{category}**")
                for i, (name, path) in enumerate(item_categories[category]):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(path if os.path.exists(path) else np.zeros((60, 60, 3), dtype=np.uint8), width=60)
                    with c2:
                        if st.button(f"Select {name}", key=f"select_{category}_{i}"):
                            st.session_state.selected_item = path
                            st.success(f"Selected: {name}")

    with col_right:
        for category in ["Goggles", "Tiaras", "Hats"]:
            if category in item_categories:
                st.write(f"**{category}**")
                for i, (name, path) in enumerate(item_categories[category]):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(path if os.path.exists(path) else np.zeros((60, 60, 3), dtype=np.uint8), width=60)
                    with c2:
                        if st.button(f"Select {name}", key=f"select_{category}_{i}"):
                            st.session_state.selected_item = path
                            st.success(f"Selected: {name}")

    
    if st.session_state.get("processed_image") is not None:
        st.subheader("📥 Download Your Try-On")
        result_pil = Image.fromarray(st.session_state.processed_image)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            result_pil.save(tmp_file.name)
            with open(tmp_file.name, "rb") as file:
                st.download_button(
                    label="Download Result",
                    data=file,
                    file_name="virtual_try_on_result.png",
                    mime="image/png",
                    key=f"download_btn_{datetime.now().timestamp()}"
                )







elif st.session_state.page == 'about':
    st.markdown('<h1 class="main-header">About VFIT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered virtual try-on technology</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## ✨ What is VFIT?
    
    VFIT is an advanced AI-powered virtual try-on application that allows you to see how different clothing items and accessories look on you without physically wearing them.
    
    ## 🔧 Technology
    
    - **Computer Vision**: Uses OpenCV for real-time face detection and image processing
    - **AI Processing**: Advanced algorithms for realistic item placement and scaling
    - **Real-time Processing**: Instant virtual try-on results through your webcam
    
    ## 🎯 Features
    
    - **Multiple Categories**: Necklaces, Earrings, Tiaras, T-shirts, and Goggles
    - **Real-time Try-on**: See results instantly through your camera
    - **Easy Navigation**: Intuitive interface for browsing and selecting items
    - **Responsive Design**: Works on desktop and mobile devices
    
    ## 🚀 How to Use
    
    1. **Browse Wardrobe**: Explore different categories of virtual items
    2. **Select Items**: Choose the items you want to try on
    3. **Use Camera**: Take a photo or use live camera feed
    4. **See Results**: View yourself wearing the virtual items instantly
    
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096;'>✨ VFIT - AI Virtual Try-On Experience</div>",
    unsafe_allow_html=True
)
