import streamlit as st
import cv2
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import tempfile
from datetime import datetime
import uuid
import dlib
import math

# Page configuration
st.set_page_config(
    page_title="✨ VFIT - AI Virtual Try-On",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
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

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'cart_items' not in st.session_state:
    st.session_state.cart_items = []
if 'current_item' not in st.session_state:
    st.session_state.current_item = None

# Initialize face detector and predictor
@st.cache_resource
def load_face_detector():
    try:
        # Try to load dlib's face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        
        # Try multiple possible paths for the shape predictor
        predictor_paths = [
            'shape_predictor_68_face_landmarks.dat',
            'data/shape_predictor_68_face_landmarks.dat',
            './data/shape_predictor_68_face_landmarks.dat'
        ]
        
        for path in predictor_paths:
            if os.path.exists(path):
                predictor = dlib.shape_predictor(path)
                st.success(f"Loaded shape predictor from: {path}")
                return detector, predictor
        
        # Fallback to Haar cascade if dlib's model is not found
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            st.warning("Using Haar cascade as fallback. For better results, download shape_predictor_68_face_landmarks.dat")
            return face_cascade, None
        
        st.error("No face detection model found. Please ensure shape_predictor_68_face_landmarks.dat is in the data/ directory.")
        return None, None
        
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None, None

def get_face_landmarks(face_detector, predictor, frame, face_rect=None):
    """Detect facial landmarks using dlib's predictor"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # If using dlib's HOG detector
    if predictor is not None:
        if face_rect is None:
            # If no face_rect provided, detect faces
            faces = face_detector(gray, 1)
            if len(faces) == 0:
                return None
            face_rect = faces[0]  # Use first detected face
            
        # Get landmarks
        landmarks = predictor(gray, face_rect)
        return [(p.x, p.y) for p in landmarks.parts()]
    
    # Fallback for Haar cascade
    elif face_rect is not None:
        x, y, w, h = face_rect
        # Return approximate landmark positions based on face rectangle
        return [
            (x + w//2, y + h//3),       # Nose tip
            (x, y + h//3),              # Left eye
            (x + w, y + h//3),          # Right eye
            (x + w//4, y + h//2),       # Left mouth corner
            (x + 3*w//4, y + h//2),     # Right mouth corner
            (x + w//2, y),              # Forehead
            (x + w//2, y + h)           # Chin
        ]
    
    return None

def calculate_face_angle(landmarks):
    """Calculate the rotation angle of the face based on eye positions"""
    if len(landmarks) < 3:  # Need at least eyes and nose
        return 0
    
    # Use eye positions to determine angle
    left_eye = np.array(landmarks[0] if isinstance(landmarks[0], (tuple, list)) else (landmarks[0].x, landmarks[0].y))
    right_eye = np.array(landmarks[1] if isinstance(landmarks[1], (tuple, list)) else (landmarks[1].x, landmarks[1].y))
    
    # Calculate angle between eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    return angle

def rotate_image(image, angle, center=None, scale=1.0):
    """Rotate an image around its center"""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    # Perform the rotation and return the image
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Process frame for virtual try-on
def process_frame(frame, item_path):
    # Load detector and predictor
    detector, predictor = load_face_detector()
    if detector is None:
        return frame
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    if predictor is not None:  # Using dlib
        faces = detector(gray, 1)
        if len(faces) == 0:
            return None
        face_rect = faces[0]  # Use first detected face
        landmarks = get_face_landmarks(detector, predictor, frame, face_rect)
        
        # Get face bounding box from landmarks
        if landmarks:
            x = min(p[0] for p in landmarks)
            y = min(p[1] for p in landmarks)
            w = max(p[0] for p in landmarks) - x
            h = max(p[1] for p in landmarks) - y
        else:
            # Fallback to face rectangle
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    else:  # Using Haar cascade
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        landmarks = get_face_landmarks(detector, predictor, frame, (x, y, w, h))
    
    # Load the item image with transparency
    if not os.path.exists(item_path):
        st.error(f"Item image not found: {item_path}")
        return frame
    
    item_img = cv2.imread(item_path, cv2.IMREAD_UNCHANGED)
    if item_img is None:
        st.error(f"Failed to load item image: {item_path}")
        return frame
    
    # Process the frame
    result = frame.copy()
    
    # Calculate face angle for rotation compensation
    face_angle = calculate_face_angle(landmarks) if landmarks else 0
    
    # Determine placement based on item type
    if "necklace" in item_path.lower():
        # Place necklace based on chin and shoulder landmarks
        if landmarks and len(landmarks) > 8:  # Ensure we have enough landmarks
            chin = landmarks[8] if len(landmarks) > 8 else (x + w//2, y + h)
            neck_base = (chin[0], chin[1] + int(h * 0.2))
            item_width = int(w * 1.5)
            item_height = int(h * 0.4)
            x_offset = neck_base[0] - item_width // 2
            y_offset = neck_base[1]
        else:
            item_height = int(h * 0.5)
            item_width = int(w * 1.5)
            y_offset = int(y + h * 0.8)
            x_offset = int(x - w * 0.25)
            
        item_resized = cv2.resize(item_img, (item_width, item_height))
        result = overlay_image(result, item_resized, x_offset, y_offset, angle=-face_angle)
        
    elif "earring" in item_path.lower():
        # Place earrings based on ear landmarks
        if landmarks and len(landmarks) > 15:  # Ensure we have ear landmarks
            # Left ear (landmark 0 is left side of face)
            left_ear = (max(0, x - w//4), y + h//3)
            # Right ear (landmark 16 is right side of face)
            right_ear = (min(frame.shape[1], x + w + w//4), y + h//3)
            
            item_height = int(h * 0.3)
            item_width = int(h * 0.2)
            
            # Left earring
            left_earring = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, left_earring, 
                                 left_ear[0] - item_width//2, 
                                 left_ear[1] - item_height//2,
                                 angle=-face_angle)
            
            # Right earring (flipped)
            right_earring = cv2.flip(cv2.resize(item_img, (item_width, item_height)), 1)
            result = overlay_image(result, right_earring, 
                                 right_ear[0] - item_width//2, 
                                 right_ear[1] - item_height//2,
                                 angle=-face_angle)
        else:
            # Fallback to simple positioning
            item_height = int(h * 0.3)
            item_width = int(w * 0.3)
            y_offset = int(y + h * 0.3)
            
            # Left earring
            x_offset_left = max(0, x - w//3)
            item_resized = cv2.resize(item_img, (item_width, item_height))
            result = overlay_image(result, item_resized, x_offset_left, y_offset, angle=-face_angle)
            
            # Right earring (flipped)
            x_offset_right = min(frame.shape[1] - item_width, x + w - w//3)
            item_flipped = cv2.flip(item_resized, 1)
            result = overlay_image(result, item_flipped, x_offset_right, y_offset, angle=-face_angle)
    
    elif any(x in item_path.lower() for x in ["tiara", "goggle", "sunglass", "glasses"]):
        # Place on top of head/eyes
        if landmarks and len(landmarks) > 27:  # Check for eye landmarks
            # Position between eyes (landmarks 39 and 42)
            left_eye = landmarks[39] if isinstance(landmarks[39], (tuple, list)) else (landmarks[39].x, landmarks[39].y)
            right_eye = landmarks[42] if isinstance(landmarks[42], (tuple, list)) else (landmarks[42].x, landmarks[42].y)
            
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            
            item_width = int(eye_distance * 2.2)
            item_height = int(item_width * 0.4)
            
            x_offset = eye_center[0] - item_width // 2
            y_offset = eye_center[1] - item_height // 2 - int(eye_distance * 0.3)
        else:
            item_height = int(h * 0.4)
            item_width = int(w * 1.2)
            y_offset = int(y - h * 0.3)
            x_offset = int(x - w * 0.1)
        
        item_resized = cv2.resize(item_img, (item_width, item_height))
        result = overlay_image(result, item_resized, x_offset, y_offset, angle=-face_angle)
    
    elif any(x in item_path.lower() for x in ["top", "shirt", "t-shirt"]):
        # Place on body below face
        if landmarks and len(landmarks) > 8:  # Check for chin landmark
            chin = landmarks[8] if isinstance(landmarks[8], (tuple, list)) else (landmarks[8].x, landmarks[8].y)
            item_width = int(w * 1.8)
            item_height = int(h * 1.5)
            x_offset = chin[0] - item_width // 2
            y_offset = chin[1]
        else:
            item_height = int(h * 1.5)
            item_width = int(w * 1.5)
            y_offset = int(y + h * 1.2)
            x_offset = int(x - w * 0.25)
        
        item_resized = cv2.resize(item_img, (item_width, item_height))
        result = overlay_image(result, item_resized, x_offset, y_offset, angle=-face_angle)
    
    else:
        # Default placement (centered on face)
        item_height = int(h * 0.8)
        item_width = int(w * 0.8)
        y_offset = y
        x_offset = x
        
        item_resized = cv2.resize(item_img, (item_width, item_height))
        result = overlay_image(result, item_resized, x_offset, y_offset, angle=-face_angle)
    
    return result

# Helper function to overlay images with transparency
def overlay_image(background, overlay, x, y, angle=0, scale=1.0):
    """Overlay an image with transparency onto a background image with optional rotation and scaling"""
    # If the overlay doesn't have an alpha channel, add one
    if overlay.shape[2] < 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    # Rotate and scale the overlay if needed
    if angle != 0 or scale != 1.0:
        h, w = overlay.shape[0], overlay.shape[1]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    # Get the dimensions of the overlay
    h, w = overlay.shape[0], overlay.shape[1]
    
    # Calculate the region of interest in the background
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    
    # If the overlay is completely outside the background, return the original
    if y1 >= y2 or x1 >= x2:
        return background
    
    # Calculate the corresponding region in the overlay
    overlay_y1 = max(0, -y)
    overlay_x1 = max(0, -x)
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_x2 = overlay_x1 + (x2 - x1)
    
    # Extract the alpha channel and create a mask
    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=-1)  # Add channel dimension for broadcasting
    alpha_inv = 1.0 - alpha
    
    # Blend the images using the alpha channel
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (
            alpha.squeeze() * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
            alpha_inv.squeeze() * background[y1:y2, x1:x2, c]
        )
        roi = (overlay_colors * alpha_mask + roi * (1 - alpha_mask)).astype(np.uint8)
    else:
        # If no alpha channel, simply overlay
        roi = overlay_portion
    
    # Put the modified ROI back into the background
    background[roi_y1:roi_y2, roi_x1:roi_x2] = roi
    
    return background

# Function to save uploaded file
def save_uploaded_file(uploaded_file, category):
    try:
        # Create directory if it doesn't exist
        os.makedirs(f"static/images/uploaded_{category}", exist_ok=True)
        
        # Generate a unique filename to avoid conflicts
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(f"static/images/uploaded_{category}", unique_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path, unique_filename
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None, None

# Function to delete uploaded file
def delete_uploaded_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False

# Sidebar navigation
st.sidebar.title("✨ VFIT")

# Upload new item section in sidebar
with st.sidebar.expander("➕ Add New Item"):
    st.subheader("Upload New Item")
    item_name = st.text_input("Item Name")
    item_category = st.selectbox("Category", ["Necklaces", "Earrings", "Tiaras", "Goggles", "Tops"])
    uploaded_file = st.file_uploader("Upload Item Image", type=["png", "jpg", "jpeg"])
    
    if st.button("Add to Wardrobe") and uploaded_file is not None and item_name:
        file_path, filename = save_uploaded_file(uploaded_file, item_category.lower())
        if file_path and filename:
            st.success(f"Successfully added {item_name} to {item_category} category!")
            # Add to session state to show immediately
            if f'uploaded_{item_category.lower()}' not in st.session_state:
                st.session_state[f'uploaded_{item_category.lower()}'] = []
            st.session_state[f'uploaded_{item_category.lower()}'].append({
                'name': item_name,
                'path': file_path,
                'filename': filename,
                'id': str(uuid.uuid4())
            })
        else:
            st.error("Failed to save the item. Please try again.")

page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Home", "👔 Wardrobe", "📸 Try-On", "ℹ️ About"]
)

# Update session state based on selection
if page == "🏠 Home":
    st.session_state.page = 'home'
elif page == "👔 Wardrobe":
    st.session_state.page = 'wardrobe'
elif page == "📸 Try-On":
    st.session_state.page = 'tryon'
elif page == "ℹ️ About":
    st.session_state.page = 'about'

# Home Page
if st.session_state.page == 'home':
    st.markdown('<h1 class="main-header">Welcome to VFIT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI-powered virtual try-on experience. Upload wardrobe items and profile photos, then create stunning virtual try-ons with the power of AI.</p>', unsafe_allow_html=True)
    
    # Stats section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h2>👔</h2>
            <h3>Wardrobe Items</h3>
            <h2>50+</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h2>📸</h2>
            <h3>Profile Photos</h3>
            <h2>Ready</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h2>✨</h2>
            <h3>Virtual Try-Ons</h3>
            <h2>Unlimited</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started section
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
    
    # CTA buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Start with Try-On", type="primary", use_container_width=True, key="home_tryon_btn"):
            st.session_state.page = 'tryon'
            st.rerun()

    
    with col2:
        if st.button("👔 Browse Wardrobe", use_container_width=True, key="home_wardrobe_btn"):
            st.session_state.page = 'wardrobe'
            st.rerun()

# Wardrobe Page
elif st.session_state.page == 'wardrobe':
    st.markdown('<h1 class="main-header">Virtual Wardrobe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Browse and select from our collection of virtual clothing items</p>', unsafe_allow_html=True)
    
    # Create necessary directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # Initialize uploaded items file if it doesn't exist
    uploaded_items_file = 'static/uploads/uploaded_items.json'
    if not os.path.exists(uploaded_items_file):
        with open(uploaded_items_file, 'w') as f:
            json.dump([], f)
    
    # Load uploaded items
    try:
        with open(uploaded_items_file, 'r') as f:
            uploaded_items = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        uploaded_items = []
    
    # File uploader in the sidebar
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
                    # Save the uploaded file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{item_category.lower()}_{timestamp}_{uploaded_file.name}"
                    filepath = os.path.join('static', 'uploads', filename)
                    
                    # Save the file
                    with open(filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add to uploaded items
                    new_item = {
                        'name': item_name,
                        'category': item_category,
                        'path': filepath,
                        'date_added': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    uploaded_items.append(new_item)
                    
                    # Save updated items list
                    with open(uploaded_items_file, 'w') as f:
                        json.dump(uploaded_items, f)
                    
                    st.success(f"Successfully uploaded {item_name} to your wardrobe!")
                else:
                    st.warning("Please provide both an item name and select an image file.")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👗 Necklaces", "👂 Earrings", "👑 Hats & Tiaras", "👕 T-shirts", "👓 Goggles", "📤 My Uploads"])
    
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
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
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
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
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
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
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
        
        # Add uploaded T-shirts
        for item in uploaded_items:
            if item['category'].lower() == 't-shirt':
                tshirts.append((item['name'], item['path']))
                
        for i, (name, path) in enumerate(tshirts):
            with cols[i % 3]:
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 0, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_column_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"tshirt_try_{name}_{i}"):
                    st.session_state.selected_item = path
                    st.session_state.page = 'tryon'
                    st.rerun()
    
    with tab5:
        st.markdown('<h2 class="category-header">Goggles</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        goggles = [
            ("Goggles", "static/images/Sunglasses61.png"),
            ("Sun Glasses", "static/images/Sunglasses62.png"),
            ("Spectacles", "static/images/Sunglasses63.png"),
            ("Sun Glasses", "static/images/Sunglasses64.png"),
            ("Shades", "static/images/Sunglasses65.png"),
            ("Shades", "static/images/Sunglasses66.png"),
        ]
        
        for i, (name, path) in enumerate(goggles):
            with cols[i % 3]:
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 0, 255), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                
                if st.button(f"Try {name}", key=f"goggle_try_{name}_{i}"):
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
                            
                            # Add buttons with better styling
                            st.markdown("""
                            <style>
                                .delete-btn {
                                    background-color: #ff4b4b !important;
                                    color: white !important;
                                    border: none !important;
                                    padding: 0.5rem 1rem !important;
                                    border-radius: 4px !important;
                                    font-weight: bold !important;
                                    margin-top: 0.5rem !important;
                                    width: 100% !important;
                                }
                                .tryon-btn {
                                    background-color: #4CAF50 !important;
                                    color: white !important;
                                    border: none !important;
                                    padding: 0.5rem 1rem !important;
                                    border-radius: 4px !important;
                                    font-weight: bold !important;
                                    margin-top: 0.5rem !important;
                                    width: 100% !important;
                                }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Try on button
                            if st.button(f"👗 Try on {item['name']}", 
                                       key=f"uploaded_try_{item['category']}_{i}_{item['name'].replace(' ', '_')}",
                                       help=f"Try on {item['name']}"):
                                st.session_state.selected_item = item['path']
                                st.session_state.page = 'tryon'
                                st.rerun()
                            
                            # Delete button with confirmation
                            if st.button(f"🗑️ Delete {item['name']}", 
                                       key=f"delete_{item['category']}_{i}_{item['name'].replace(' ', '_')}",
                                       help=f"Delete {item['name']} from your wardrobe"):
                                # Show confirmation dialog
                                if st.session_state.get(f'confirm_delete_{i}') == item['path']:
                                    # Delete the file
                                    if os.path.exists(item['path']):
                                        os.remove(item['path'])
                                    # Remove from uploaded items list
                                    uploaded_items.remove(item)
                                    # Save the updated list
                                    with open(uploaded_items_file, 'w') as f:
                                        json.dump(uploaded_items, f)
                                    st.success(f"Successfully deleted {item['name']}")
                                    st.rerun()
                                else:
                                    st.session_state[f'confirm_delete_{i}'] = item['path']
                                    st.warning(f"Are you sure you want to delete {item['name']}? Click the delete button again to confirm.")
                                    st.experimental_rerun()
                        else:
                            st.warning(f"Could not find image: {item['path']}")
                    except Exception as e:
                        st.error(f"Error loading {item.get('name', 'item')}: {str(e)}")
        st.markdown('<h2 class="category-header">Goggles</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        goggles = [
            ("Goggles", "static/images/Sunglasses61.png"),
            ("Sun Glasses", "static/images/Sunglasses62.png"),
            ("Spectacles", "static/images/Sunglasses63.png"),
            ("Sun Glasses", "static/images/Sunglasses64.png"),
            ("Shades", "static/images/Sunglasses65.png"),
            ("Shades", "static/images/Sunglasses66.png"),
        ]
        
        for i, (name, path) in enumerate(goggles):
            with cols[i % 3]:
                # Create placeholder if image doesn't exist
                if not os.path.exists(path):
                    # Create a placeholder image
                    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.rectangle(placeholder, (50, 50), (150, 150), (0, 255, 0), -1)
                    cv2.putText(placeholder, name, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    st.image(placeholder, caption=name, use_container_width=True)
                else:
                    st.image(path, caption=name, use_container_width=True)
                    
                unique_key = f"goggle_try_{name}_{i}_{uuid.uuid4().hex}"
                if st.button(f"Try {name}", key=unique_key):
                    st.session_state.selected_item = path
                    st.session_state.page = "tryon"
                    st.rerun()

# Try-On Page
elif st.session_state.page == 'tryon':
    st.markdown('<h1 class="main-header">Virtual Try-On</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Use your webcam to try on virtual items in real-time</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Camera Feed")
        
        # Camera input with error handling
        try:
            camera_input = st.camera_input("Take a photo to try on items")
            
            if camera_input is not None:
                # Convert the uploaded image to OpenCV format
                image = Image.open(camera_input)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Get selected items or default
                items_to_try = []
                if st.session_state.current_item:
                    items_to_try = [st.session_state.current_item]
                elif st.session_state.cart_items:
                    items_to_try = [item['path'] for item in st.session_state.cart_items]
                else:
                    items_to_try = ['static/images/Necklace11.png']  # Default item
                
                # Process the frame with virtual try-on
                try:
                    processed_frame = frame.copy()
                    for item_path in items_to_try:
                        processed_frame = process_frame(processed_frame, item_path)
                    
                    if processed_frame is None:
                        processed_frame = frame.copy()  # Fallback to original if processing fails
                    
                    if processed_frame is not None:
                        # Convert back to RGB for display
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st.image(processed_frame_rgb, caption="Virtual Try-On Result", use_container_width=True)
                        st.session_state.processed_image = processed_frame_rgb
                    else:
                        st.image(frame, caption="Original Photo", use_container_width=True)
                        st.warning("No face detected. Please ensure your face is clearly visible in the photo.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.image(frame, caption="Original Photo", use_container_width=True)
                    
        except Exception as e:
            st.error("Error accessing webcam. Please ensure your camera is connected and permissions are granted.")
            st.info("As an alternative, you can upload an image below.")
            uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Get selected items or default
                items_to_try = []
                if st.session_state.current_item:
                    items_to_try = [st.session_state.current_item]
                elif st.session_state.cart_items:
                    items_to_try = [item['path'] for item in st.session_state.cart_items]
                else:
                    items_to_try = ['static/images/Necklace11.png']  # Default item
                
                # Process the frame with virtual try-on
                try:
                    processed_frame = frame.copy()
                    for item_path in items_to_try:
                        processed_frame = process_frame(processed_frame, item_path)
                    
                    if processed_frame is None:
                        processed_frame = frame.copy()  # Fallback to original if processing fails
                    
                    if processed_frame is not None:
                        # Convert back to RGB for display
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st.image(processed_frame_rgb, caption="Virtual Try-On Result", use_container_width=True)
                        st.session_state.processed_image = processed_frame_rgb
                    else:
                        st.image(frame, caption="Original Photo", use_container_width=True)
                        st.warning("No face detected. Please ensure your face is clearly visible in the photo.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.image(frame, caption="Original Photo", use_container_width=True)
    
    with col2:
        st.subheader("👔 Select Item")
        
        # Base item categories
        item_categories = {
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
                ("Tiara", "static/images/Tiaras31.png"),
                ("Tiara", "static/images/Tiaras32.png"),
                ("Tiara", "static/images/Tiaras33.png"),
            ],
            "Goggles": [
                ("Goggles", "static/images/Sunglasses61.png"),
                ("Sun Glasses", "static/images/Sunglasses62.png"),
                ("Spectacles", "static/images/Sunglasses63.png"),
            ],
            "Tops": [
                ("Orange T-Shirt", "static/images/Tops41.png"),
                ("Pink T-Shirt", "static/images/Tops42.png"),
                ("White T-Shirt", "static/images/Tops43.png")
            ]
        }
        
        # Add uploaded items to their respective categories
        for category in ["necklaces", "earrings", "tiaras", "goggles", "tops"]:
            session_key = f'uploaded_{category}'
            if session_key in st.session_state and st.session_state[session_key]:
                category_name = category.capitalize()
                if category_name not in item_categories:
                    item_categories[category_name] = []
                # Add only items that aren't already in the list
                for item in st.session_state[session_key]:
                    item_tuple = (item['name'], item['path'])
                    if item_tuple not in item_categories[category_name]:
                        item_categories[category_name].append(item_tuple)
                        
                        # Add delete button for uploaded items
                        if st.button(f"🗑️ Delete {item['name']}", key=f"delete_{item['id']}"):
                            if delete_uploaded_file(item['path']):
                                st.session_state[session_key] = [x for x in st.session_state[session_key] if x['id'] != item['id']]
                                st.success(f"Successfully deleted {item['name']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete the item. Please try again.")
        
        for category, items in item_categories.items():
            st.write(f"**{category}**")
            for name, path in items:
                col_img, col_btn = st.columns([1, 2])
                with col_img:
                    # Create placeholder if image doesn't exist
                    if not os.path.exists(path):
                        # Create a placeholder image
                        placeholder = np.zeros((60, 60, 3), dtype=np.uint8)
                        if "necklace" in category.lower():
                            color = (0, 255, 255)
                        elif "earring" in category.lower():
                            color = (255, 0, 255)
                        elif "tiara" in category.lower():
                            color = (255, 255, 0)
                        elif "goggle" in category.lower():
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                        cv2.rectangle(placeholder, (10, 10), (50, 50), color, -1)
                        st.image(placeholder, width=60)
                    else:
                        st.image(path, width=60)
                with col_btn:
                    if st.button(f"Add to Cart", key=f"add_{path}"):
                        if path not in [item['path'] for item in st.session_state.cart_items]:
                            st.session_state.cart_items.append({
                                'name': name,
                                'path': path,
                                'id': str(uuid.uuid4())  # Unique ID for each item
                            })
                            st.success(f"Added {name} to cart!")
                        else:
                            st.warning("Item already in cart")
        
        # Shopping Cart
        st.subheader("🛒 Shopping Cart")
        
        if not st.session_state.cart_items:
            st.info("Your cart is empty. Add items from above.")
        else:
            for i, item in enumerate(st.session_state.cart_items):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.image(item['path'], width=50)
                with col2:
                    st.write(f"**{item['name']}**")
                with col3:
                    if st.button("❌", key=f"remove_{item['id']}"):
                        st.session_state.cart_items = [x for x in st.session_state.cart_items if x['id'] != item['id']]
                        st.rerun()
            
            # Try on all items button
            if st.button("👕 Try On All Items"):
                st.session_state.current_item = None  # Reset current item
                st.session_state.page = "tryon"
                st.rerun()
        
        # Current selection for single item try-on
        st.subheader("👔 Quick Try-On")
        if 'selected_item' in st.session_state and st.session_state.selected_item is not None:
            st.write("**Currently Selected:**")
            st.write(st.session_state.selected_item.split('/')[-1])
            if st.button("Try On This Item"):
                st.session_state.current_item = st.session_state.selected_item
                st.session_state.page = "tryon"
                st.rerun()
        else:
            st.write("**No item selected**")
        
        # Download button for processed image
        if st.session_state.processed_image is not None:
            st.subheader("📥 Download Your Try-On")
            # Convert to PIL Image for download
            result_pil = Image.fromarray(st.session_state.processed_image)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                result_pil.save(tmp_file.name)
                
                # Create download button
                with open(tmp_file.name, "rb") as file:
                    btn = st.download_button(
                        label="Download Result",
                        data=file,
                        file_name="virtual_try_on_result.png",
                        mime="image/png"
                    )

# About Page
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
    
    ---
    
    Made with ❤️ using Streamlit and OpenCV
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096;'>✨ VFIT - AI Virtual Try-On Experience</div>",
    unsafe_allow_html=True
)
