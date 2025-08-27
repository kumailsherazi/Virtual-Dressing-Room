import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import virtual_trial

# Page configuration
st.set_page_config(
    page_title="✨ Miroir - AI Virtual Try-On",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar navigation
st.sidebar.title("✨ Miroir")
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
    st.markdown('<h1 class="main-header">Welcome to Miroir</h1>', unsafe_allow_html=True)
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
        if st.button("🎯 Start with Try-On", type="primary", use_container_width=True):
            st.session_state.page = 'tryon'
            st.rerun()
    
    with col2:
        if st.button("👔 Browse Wardrobe", use_container_width=True):
            st.session_state.page = 'wardrobe'
            st.rerun()

# Wardrobe Page
elif st.session_state.page == 'wardrobe':
    st.markdown('<h1 class="main-header">Virtual Wardrobe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Browse and select from our collection of virtual clothing items</p>', unsafe_allow_html=True)
    
    # Category tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💎 Necklaces", "💍 Earrings", "👑 Tiaras", "👕 T-shirts", "🕶️ Goggles"])
    
    with tab1:
        st.markdown('<h2 class="category-header">Necklaces</h2>', unsafe_allow_html=True)
        cols = st.columns(3)
        necklaces = [
            ("Silver Necklace", "static/images/Necklace11.png"),
            ("Gold Necklace", "static/images/Necklace12.png"),
            ("Silver Necklace", "static/images/Necklace14.png"),
            ("Thread Necklace", "static/images/Necklace15.png"),
            ("Gold Chain", "static/images/Necklace16.png"),
            ("Gold Chain", "static/images/Necklace17.png"),
        ]
        
        for i, (name, path) in enumerate(necklaces):
            with cols[i % 3]:
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Try {name}", key=f"necklace_{i}"):
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
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Try {name}", key=f"earring_{i}"):
                        st.session_state.selected_item = path
                        st.session_state.page = 'tryon'
                        st.rerun()
    
    with tab3:
        st.markdown('<h2 class="category-header">Tiaras</h2>', unsafe_allow_html=True)
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
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Try {name}", key=f"tiara_{i}"):
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
            ("White T-Shirt", "static/images/Tops47.jpg"),
            ("Black T-Shirt", "static/images/Tops48.png"),
            ("White and Black Full Sleeves", "static/images/Tops49.png"),
        ]
        
        for i, (name, path) in enumerate(tshirts):
            with cols[i % 3]:
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Try {name}", key=f"tshirt_{i}"):
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
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Try {name}", key=f"goggle_{i}"):
                        st.session_state.selected_item = path
                        st.session_state.page = 'tryon'
                        st.rerun()

# Try-On Page
elif st.session_state.page == 'tryon':
    st.markdown('<h1 class="main-header">Virtual Try-On</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Use your webcam to try on virtual items in real-time</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Camera Feed")
        
        # Camera input
        camera_input = st.camera_input("Take a photo to try on items")
        
        if camera_input is not None:
            # Convert the uploaded image to OpenCV format
            image = Image.open(camera_input)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get selected item or default
            selected_item = st.session_state.get('selected_item', 'static/images/Necklace11.png')
            
            # Process the frame with virtual try-on
            try:
                processed_frame = virtual_trial.process_frame(frame, selected_item)
                
                if processed_frame is not None:
                    # Convert back to RGB for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(processed_frame_rgb, caption="Virtual Try-On Result", use_container_width=True)
                else:
                    st.image(frame, caption="Original Photo", use_container_width=True)
                    st.warning("No face detected. Please ensure your face is clearly visible in the photo.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.image(frame, caption="Original Photo", use_column_width=True)
    
    with col2:
        st.subheader("👔 Select Item")
        
        # Quick item selection
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
            ]
        }
        
        for category, items in item_categories.items():
            st.write(f"**{category}**")
            for name, path in items:
                if os.path.exists(path):
                    col_img, col_btn = st.columns([1, 2])
                    with col_img:
                        st.image(path, width=80)
                    with col_btn:
                        if st.button(f"Select {name}", key=f"select_{path}"):
                            st.session_state.selected_item = path
                            st.success(f"Selected: {name}")
        
        # Current selection
        if 'selected_item' in st.session_state:
            st.write("**Currently Selected:**")
            st.write(st.session_state.selected_item.split('/')[-1])

# About Page
elif st.session_state.page == 'about':
    st.markdown('<h1 class="main-header">About Miroir</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered virtual try-on technology</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## ✨ What is Miroir?
    
    Miroir is an advanced AI-powered virtual try-on application that allows you to see how different clothing items and accessories look on you without physically wearing them.
    
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
    "<div style='text-align: center; color: #718096;'>✨ Miroir - AI Virtual Try-On Experience</div>",
    unsafe_allow_html=True
)
