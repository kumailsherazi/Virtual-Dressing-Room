import cv2, os
import math
import numpy as np
from PIL import Image

ACTIVE_IMAGES = [0 for i in range(10)]
SPRITES = [0 for i in range(10)]
IMAGES = {i: [] for i in range(10)}
PHOTOS = {i: [] for i in range(10)}

# Function to rotate image without cropping
def rotate_bound(image, angle):
    # Get image dimensions and calculate the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Compute new dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # Perform rotation
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def initialize_images_and_photos(file_path):
    global IMAGES, PHOTOS
    
    if not file_path:
        print("Error: No file path provided")
        return
        
    # Convert path to use forward slashes for consistency
    file_path = file_path.replace('\\', '/')
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    if not filename:
        print(f"Error: Could not extract filename from path: {file_path}")
        return
    
    # Try to extract index from filename (e.g., 'Hat1.png' -> 1)
    try:
        # Extract the first number found in the filename
        numbers = [int(s) for s in filename if s.isdigit()]
        if not numbers:
            print(f"Warning: No numbers found in filename: {filename}")
            idx = 0  # Default to 0 if no number found
        else:
            idx = numbers[0]  # Use the first number found
        
        # Check if file exists, if not try to find it in static/images
        if not os.path.isfile(file_path):
            # Try to find the file in static/images
            static_path = os.path.join('static', 'images', filename).replace('\\', '/')
            if os.path.isfile(static_path):
                file_path = static_path
                print(f"Found file at: {file_path}")
            else:
                # Try to find any file with a similar pattern
                print(f"Warning: Could not find file: {filename}")
                # Look for any file that starts with the same prefix (e.g., 'Hat')
                prefix = ''.join([c for c in filename if not c.isdigit()]).split('.')[0]
                if prefix:
                    image_dir = os.path.join('static', 'images')
                    if os.path.exists(image_dir):
                        for f in os.listdir(image_dir):
                            if f.startswith(prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                file_path = os.path.join(image_dir, f).replace('\\', '/')
                                print(f"Using alternative file: {file_path}")
                                break
                
                if not os.path.isfile(file_path):
                    print(f"Error: Could not find any suitable file for: {filename}")
                    return
        
        # Read and process the image
        sprite_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if sprite_image is not None:
            # If image doesn't have alpha channel, add one
            if sprite_image.shape[2] == 3:
                alpha_channel = np.ones((sprite_image.shape[0], sprite_image.shape[1]), dtype=sprite_image.dtype) * 255
                sprite_image = cv2.merge((sprite_image, alpha_channel))
                
            IMAGES[idx].append(sprite_image)
            photo = cv2.resize(sprite_image, (150, 100))
            PHOTOS[idx].append(photo) if idx in PHOTOS else PHOTOS.update({idx: [photo]})
            print(f"Successfully loaded image: {os.path.basename(file_path)}")
        else:
            print(f"Error: Could not read image: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def put_sprite(num, k):
    global SPRITES, ACTIVE_IMAGES
    SPRITES[num] = 1
    ACTIVE_IMAGES[num] = k if SPRITES[num] else None

def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])
    
    # Ensure the sprite doesn't go beyond frame boundaries
    if y_offset + h >= imgH: 
        sprite = sprite[0:imgH - y_offset, :, :]
    if x_offset + w >= imgW: 
        sprite = sprite[:, 0:imgW - x_offset, :]
    if x_offset < 0: 
        sprite = sprite[:, abs(x_offset)::, :]
        w = sprite.shape[1]
        x_offset = 0
    if y_offset < 0:
        sprite = sprite[abs(y_offset)::, :, :]
        h = sprite.shape[0]
        y_offset = 0
        
    # Extract the alpha channel from the sprite
    if sprite.shape[2] == 4:
        alpha = sprite[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Extract the color channels
        sprite_rgb = sprite[:, :, :3]
        
        # Blend the sprite with the frame
        for c in range(3):
            frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
                sprite_rgb[:, :, c] * alpha[:, :, 0] + \
                frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1.0 - alpha[:, :, 0])
    else:
        # If no alpha channel, just overlay the sprite
        frame[y_offset:y_offset + h, x_offset:x_offset + w, :] = sprite
        
    return frame

def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    factor = 1.0 * head_width / sprite.shape[1]
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    y_orig = head_ypos - sprite.shape[0] if ontop else head_ypos
    if y_orig < 0: 
        sprite = sprite[abs(y_orig)::, :, :]
    return sprite, max(y_orig, 0)

def apply_Haar_filter(img, haar_cascade, scaleFact=1.05, minNeigh=3, minSizeW=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return haar_cascade.detectMultiScale(gray, scaleFactor=scaleFact, minNeighbors=minNeigh, minSize=(minSizeW, minSizeW), flags=cv2.CASCADE_SCALE_IMAGE)

def get_face_boundbox(face_rect, face_part):
    # Simplified face regions based on face rectangle for OpenCV
    x, y, w, h = face_rect
    if face_part == 1:  # eyebrows
        return x + w//4, y + h//4, w//2, h//8
    elif face_part == 6:  # mouth area
        return x + w//4, y + 3*h//4, w//2, h//4
    elif face_part == 7:  # left ear area
        return x - w//8, y + h//3, w//4, h//3
    elif face_part == 8:  # right ear area
        return x + 7*w//8, y + h//3, w//4, h//3
    else:
        return x, y, w, h

def calculate_inclination(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 == x1:  # Avoid division by zero
        return 90 if y2 > y1 else -90
    return 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))

def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:, 0])
    y = min(list_coordinates[:, 1])
    w = max(list_coordinates[:, 0]) - x
    h = max(list_coordinates[:, 1]) - y
    return x, y, w, h

def get_category_number(file_path):
    if not file_path:
        return -1
        
    parts = file_path.split('/')
    category_part = parts[-1]
    if category_part:
        digits = ''.join(filter(str.isdigit, category_part))
        if digits:
            val = int(digits)
            val = (val // 10) % 10
            return val
    return -1

def get_k(file_path):
    if not file_path:
        return -1
        
    parts = file_path.split('/')
    category_part = parts[-1]
    if category_part:
        digits = ''.join(filter(str.isdigit, category_part))
        if digits:
            val = int(digits)
            val = val % 10
            return val
    return -1

def apply_sprite(frame, sprite, w, x, y, angle, ontop=True):
    if sprite is None or sprite.size == 0:
        return frame
        
    try:
        sprite = rotate_bound(sprite, angle)
        (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
        frame = draw_sprite(frame, sprite, x, y_final)
        return frame
    except Exception as e:
        print(f"Error applying sprite: {str(e)}")
        return frame

@st.cache_resource
def load_cascade_classifier():
    try:
        # Try multiple possible paths for the cascade file
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            './haarcascade_frontalface_default.xml',
            os.path.join('data', 'haarcascade_frontalface_default.xml'),
            os.path.join('data', 'data', 'haarcascade_frontalface_default.xml')
        ]
        
        face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    print(f"Loaded cascade from: {path}")
                    return face_cascade
        
        # If not found, use alternative method
        print("Haar cascade file not found. Using alternative detection method.")
        return None
    except Exception as e:
        print(f"Error loading cascade classifier: {str(e)}")
        return None

def process_frame(frame, file_path):
    global SPRITES, ACTIVE_IMAGES, IMAGES
    sprite_applied = False
    
    if frame is None:
        print("Error: No frame provided")
        return None
        
    if isinstance(frame, bytes):
        try:
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Error: Could not decode frame data")
                return None
        except Exception as e:
            print(f"Error processing frame data: {str(e)}")
            return None
    
    # Ensure file_path is a string and handle path formatting
    if not isinstance(file_path, str) or not file_path.strip():
        print("Error: Invalid file path")
        return frame  # Return original frame if no valid path provided
    
    # Clean up the path
    file_path = file_path.strip().replace('\\', '/')
    
    # If the file doesn't exist, try to find it in static/images
    if not os.path.isfile(file_path):
        # Extract filename from path
        filename = os.path.basename(file_path)
        if not filename:
            print(f"Error: Could not extract filename from path: {file_path}")
            return frame  # Return original frame if no valid filename
            
        # Try to find the file in static/images
        static_path = os.path.join('static', 'images', filename).replace('\\', '/')
        if os.path.isfile(static_path):
            file_path = static_path
            print(f"Found file at: {file_path}")
        else:
            # Try to find any file with a similar pattern
            print(f"Warning: Could not find file: {filename}")
            prefix = ''.join([c for c in filename if not c.isdigit()]).split('.')[0]
            if prefix:
                image_dir = os.path.join('static', 'images')
                if os.path.exists(image_dir):
                    for f in os.listdir(image_dir):
                        if f.startswith(prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(image_dir, f).replace('\\', '/')
                            print(f"Using alternative file: {file_path}")
                            break
            
            if not os.path.isfile(file_path):
                print(f"Error: Could not find any suitable file for: {filename}")
                return frame  # Return original frame if no file found
    
    if frame is None or not hasattr(frame, 'shape'):
        print("Invalid frame")
        return None
    
    initialize_images_and_photos(file_path)

    # Load cascade classifier
    face_cascade = load_cascade_classifier()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = []
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    else:
        # Fallback: use a simple face detection approximation
        height, width = gray.shape
        faces = [(int(width/4), int(height/4), int(width/2), int(height/3))]
    
    for (x, y, w, h) in faces: 
        # Simplified face processing without detailed landmarks
        incl = 0  # No rotation calculation without landmarks
        index = get_category_number(file_path)
        k = get_k(file_path)
        put_sprite(index, k)
        
        if SPRITES[3]:  # Tiara
            frame = apply_sprite(frame, IMAGES[3][ACTIVE_IMAGES[3]] if len(IMAGES[3]) > ACTIVE_IMAGES[3] else None, 
                               w+45, x-20, y+20, incl, ontop=True)
            sprite_applied = True

        # Necklaces
        if SPRITES[1]:
            (x1, y1, w1, h1) = get_face_boundbox((x, y, w, h), 6)
            if len(IMAGES[1]) > ACTIVE_IMAGES[1]:
                frame = apply_sprite(frame, IMAGES[1][ACTIVE_IMAGES[1]], w1, x1, y1+125, incl, ontop=False)
                sprite_applied = True
        
        # Goggles
        if SPRITES[6]:
            (x3, y3, _, h3) = get_face_boundbox((x, y, w, h), 1)
            if len(IMAGES[6]) > ACTIVE_IMAGES[6]:
                frame = apply_sprite(frame, IMAGES[6][ACTIVE_IMAGES[6]], w, x, y3-5, incl, ontop=False)
                sprite_applied = True

        # Earrings
        if SPRITES[2]:
            if len(IMAGES[2]) > ACTIVE_IMAGES[2]:
                (x3, y3, w3, h3) = get_face_boundbox((x, y, w, h), 7)  # left ear
                frame = apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]], w3, x3-40, y3+30, incl, ontop=False)
                
                (x3, y3, w3, h3) = get_face_boundbox((x, y, w, h), 8)  # right ear
                frame = apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]], w3, x3+40, y3+75, incl, ontop=False)
                sprite_applied = True

        # Tops
        if SPRITES[4]:
            if len(IMAGES[4]) > ACTIVE_IMAGES[4]:
                # Adjust T-shirt positioning to better fit the body
                tshirt_width = int(w * 2.5)  # Make T-shirt wider than the face
                tshirt_x = x - (tshirt_width - w) // 2  # Center the T-shirt
                tshirt_y = y + h  # Position below the face
                
                # Apply the T-shirt with adjusted size and position
                frame = apply_sprite(frame, IMAGES[4][ACTIVE_IMAGES[4]], 
                                    tshirt_width, tshirt_x, tshirt_y, 
                                    incl, ontop=False)
                sprite_applied = True
            
    # Reset sprites for next frame
    SPRITES = [0 for i in range(10)]
    
    return frame if sprite_applied else frame
