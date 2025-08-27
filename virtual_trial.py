import cv2, os
from imutils import rotate_bound
import math
import numpy as np

# Global state
ACTIVE_IMAGES = [0 for _ in range(10)]
SPRITES = [0 for _ in range(10)]
IMAGES = {i: [] for i in range(10)}
PHOTOS = {i: [] for i in range(10)}

def initialize_images_and_photos(file_path):
    global IMAGES, PHOTOS
    
    if not file_path:
        print("Error: No file path provided")
        return
    
    file_path = file_path.replace('\\', '/')
    filename = os.path.basename(file_path)
    if not filename:
        print(f"Error: Could not extract filename from path: {file_path}")
        return
    
    try:
        numbers = [int(s) for s in filename if s.isdigit()]
        idx = numbers[0] if numbers else 0

        if not os.path.isfile(file_path):
            static_path = os.path.join('static', 'images', filename).replace('\\', '/')
            if os.path.isfile(static_path):
                file_path = static_path
                print(f"Found file at: {file_path}")
            else:
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
            return

        sprite_image = cv2.imread(file_path, -1)
        if sprite_image is not None:
            IMAGES[idx].append(sprite_image)
            photo = cv2.resize(sprite_image, (150, 100))
            if idx in PHOTOS:
                PHOTOS[idx].append(photo)
            else:
                PHOTOS[idx] = [photo]
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
    if y_offset + h >= imgH: sprite = sprite[0:imgH - y_offset, :, :]
    if x_offset + w >= imgW: sprite = sprite[:, 0:imgW - x_offset, :]
    if x_offset < 0: 
        sprite = sprite[:, abs(x_offset)::, :]
        x_offset = 0
    for c in range(3):
        frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
            sprite[:, :, c] * (sprite[:, :, 3] / 255.0) + \
            frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1.0 - sprite[:, :, 3] / 255.0)
    return frame

def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    factor = 1.0 * head_width / sprite.shape[1]
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    y_orig = head_ypos - sprite.shape[0] if ontop else head_ypos
    if y_orig < 0: sprite = sprite[abs(y_orig)::, :, :]
    return sprite, max(y_orig, 0)

def apply_sprite(frame, sprite, w, x, y, angle, ontop=True):
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    frame = draw_sprite(frame, sprite, x, y_final)
    return frame

def get_face_boundbox(face_rect, face_part):
    x, y, w, h = face_rect
    if face_part == 1:  # eyebrows
        return x + w//4, y + h//4, w//2, h//8
    elif face_part == 6:  # mouth area
        return x + w//4, y + 3*h//4, w//2, h//4
    elif face_part == 7:  # left ear
        return x - w//8, y + h//3, w//4, h//3
    elif face_part == 8:  # right ear
        return x + 7*w//8, y + h//3, w//4, h//3
    else:
        return x, y, w, h

def get_category_number(file_path):
    category_part = os.path.basename(file_path)
    if category_part:
        val = int(''.join(filter(str.isdigit, category_part)) or 0)
        return (val // 10) % 10
    return -1

def get_k(file_path):
    category_part = os.path.basename(file_path)
    if category_part:
        val = int(''.join(filter(str.isdigit, category_part)) or 0)
        return val % 10
    return -1

def process_frame(frame, file_path):
    global SPRITES, ACTIVE_IMAGES, IMAGES
    if frame is None or not hasattr(frame, 'shape'):
        print("Error: Invalid frame")
        return None
    
    initialize_images_and_photos(file_path)

    # ✅ Always use OpenCV's built-in haarcascade path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Could not load Haarcascade.")
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        incl = 0
        index = get_category_number(file_path)
        k = get_k(file_path)
        put_sprite(index, k)

        # Tiara
        if SPRITES[3] and len(IMAGES[3]) > ACTIVE_IMAGES[3]:
            frame = apply_sprite(frame, IMAGES[3][ACTIVE_IMAGES[3]], w+45, x-20, y+20, incl, ontop=True)

        # Necklace
        if SPRITES[1] and len(IMAGES[1]) > ACTIVE_IMAGES[1]:
            (x1, y1, w1, h1) = get_face_boundbox((x, y, w, h), 6)
            frame = apply_sprite(frame, IMAGES[1][ACTIVE_IMAGES[1]], w1, x1, y1+125, incl, ontop=False)

        # Goggles
        if SPRITES[6] and len(IMAGES[6]) > ACTIVE_IMAGES[6]:
            (x3, y3, _, h3) = get_face_boundbox((x, y, w, h), 1)
            frame = apply_sprite(frame, IMAGES[6][ACTIVE_IMAGES[6]], w, x, y3-5, incl, ontop=False)

        # Earrings
        if SPRITES[2] and len(IMAGES[2]) > ACTIVE_IMAGES[2]:
            (x3, y3, w3, h3) = get_face_boundbox((x, y, w, h), 7)  # left
            frame = apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]], w3, x3-40, y3+30, incl, ontop=False)
            (x3, y3, w3, h3) = get_face_boundbox((x, y, w, h), 8)  # right
            frame = apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]], w3, x3+40, y3+75, incl)

        # T-shirt / Tops
        if SPRITES[4] and len(IMAGES[4]) > ACTIVE_IMAGES[4]:
            tshirt_width = int(w * 2.5)
            tshirt_x = x - (tshirt_width - w) // 2
            tshirt_y = y + h
            frame = apply_sprite(frame, IMAGES[4][ACTIVE_IMAGES[4]], tshirt_width, tshirt_x, tshirt_y, incl, ontop=False)

    return frame
