from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter
from collections import Counter
from pathlib import Path
import os
import json
import datetime
from zoneinfo import ZoneInfo
import random
import io
import base64
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from validation_handler import ValidationHandler
from userAuth_hendler import UserAuthHandler

# =============================
# Inisialisasi Flask & Folder
# =============================
load_dotenv()

app = Flask(__name__)
app.config['Application_ROOT']='setmutu'
app.secret_key = 'your-secret-key-here'  # Replace with a secure secret key

# Timezone configuration (default to Asia/Jakarta). Override with env APP_TZ.
APP_TZ = os.environ.get('APP_TZ', 'Asia/Jakarta')

def now_local():
    try:
        return datetime.datetime.now(ZoneInfo(APP_TZ))
    except Exception:
        # Fallback if zoneinfo/tz not available
        return datetime.datetime.now()

# Folder konfigurasi berbasis environment
_default_upload = Path('static') / 'uploads'
configured_path = os.environ.get('IMGPATH') or str(_default_upload)
UPLOAD_FOLDER = Path(configured_path).expanduser()
if not UPLOAD_FOLDER.is_absolute():
    UPLOAD_FOLDER = (Path(__file__).resolve().parent / UPLOAD_FOLDER).resolve()

# Buat folder utama sesuai konfigurasi
try:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    raise RuntimeError(f"Failed to create image directory at '{UPLOAD_FOLDER}': {exc}") from exc

def _image_fs_path(filename: str | Path) -> Path:
    """Return absolute filesystem path for an image filename."""
    return UPLOAD_FOLDER / Path(filename).name

def _image_url(filename: str) -> str:
    """Return route URL for serving stored images."""
    clean_name = Path(filename).name.replace('\\', '/')
    return url_for('serve_image', filename=clean_name)

# Inisialisasi validation handler dan auth handler
validation_handler = ValidationHandler()
auth_handler = UserAuthHandler()

# Label kelas untuk training (3-class)
CLASS_LABELS = ["ripe", "rotten", "unripe"]

def _add_ticket_record(file_name, class_result, place, valid_status=False, 
                      total_process_time=0.0, detection_time=0.0, classification_time=0.0):
    """Add a new record to PostgreSQL ticket table via ValidationHandler with timing info."""
    # Map legacy boolean valid_status to status string
    status = 'close' if (isinstance(valid_status, bool) and valid_status) or str(valid_status).lower() == 'true' else 'open'
    username = session.get('username', 'default_user')
    user_group = session.get('group', 'user')
    validation_handler.add_detection_result(
        file_name=file_name,
        group=user_group,
        class_result=class_result,
        status=status,
        place=place,
        username=username,
        total_process_time=total_process_time,
        detection_time=detection_time,
        classification_time=classification_time
    )

def generate_base_filename():
    """Generate a base filename without extension and place number
    Format: YYYYMMDDHHMMSSxxxxxx (xxxxxx is random hex)"""
    timestamp = now_local().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices('0123456789abcdef', k=6))
    return f"{timestamp}{random_str}"

# Path model
DET_MODEL_PATH = os.path.join('models', 'model_automl.tflite')
# gunakan model klasifikasi 3-kelas
CLS_MODEL_PATH = os.path.join('models', 'Mutu_320.tflite')

# =============================
# Load Model Deteksi & Klasifikasi
# =============================
det_interpreter = tf.lite.Interpreter(model_path=DET_MODEL_PATH)
det_interpreter.allocate_tensors()
det_input_details = det_interpreter.get_input_details()
det_output_details = det_interpreter.get_output_details()

cls_interpreter = tf.lite.Interpreter(model_path=CLS_MODEL_PATH)
cls_interpreter.allocate_tensors()
cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()

det_class_names = ['buah', 'batang']
cls_labels = ["ripe", "rotten", "unripe"]

# =============================
# Routes
# =============================

@app.route('/', methods=['GET', 'POST'])
def index():
    # Jika user sudah login, redirect ke /index
    if 'username' in session:
        return redirect(url_for('dashboard'))
    
    # Jika belum login, tampilkan halaman login
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = auth_handler.login_user(username, password)
        if user:
            session['username'] = username
            session['group'] = user.get('group', 'user')
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/index')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        group = request.form.get('group', 'user')  # Default to 'user' if not specified
        
        if auth_handler.register_user(username, password, group):
            return redirect(url_for('index'))
        else:
            return render_template('register.html', error="Registration failed. Username might already exist.")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('group', None)
    return redirect(url_for('index'))

@app.route('/update_valid_status', methods=['POST'])
def update_valid_status():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        image_id = data.get('image_id')
        place = data.get('place')
        
        if not image_id or place is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Convert place to integer if it's not already
        try:
            place = int(place)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid place value'}), 400

        # Use validation handler to update status
        success, message = validation_handler.update_valid_status(image_id, place)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/set_status', methods=['POST'])
def set_status():
    """Set status open/close for all entries of a base image; only uploader can change."""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    try:
        data = request.get_json() or {}
        base_name = data.get('base_name')  # e.g., 20250909125957ca6e5f
        new_status = data.get('status')  # 'open' | 'close'
        if not base_name or new_status not in ('open','close'):
            return jsonify({'error': 'Missing or invalid parameters'}), 400

        ok, msg = validation_handler.set_status_for_base(
            base_name=base_name,
            group=session.get('group', 'user'),
            new_status=new_status,
            uploader_username=session.get('username')
        )
        if ok:
            return jsonify({'success': True})
        return jsonify({'error': msg}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/submit_validations', methods=['POST'])
def submit_validations():
    """Accept a list of validations from validate.html and append to validate.txt.
    Each item includes: file_name, group, class_result, place, username, timestamp, valid_status (bool)."""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    try:
        payload = request.get_json() or {}
        items = payload.get('items', [])
        if not isinstance(items, list) or not items:
            return jsonify({'error': 'No validation items provided'}), 400

        ok, result = validation_handler.save_validations(items, session.get('username'), session.get('group','user'))
        if not ok:
            return jsonify({'error': f'Failed to save validations: {result}'}), 500
        return jsonify({'success': True, 'count': result})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/verify', methods=['POST'])
def verify_classification():
    """Route untuk verify/mengubah klasifikasi crop"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        crop_path = data.get('crop_path')
        verified_label = _normalize_label(data.get('verified_label'))
        
        if not crop_path or not verified_label:
            return jsonify({'error': 'Missing required parameters: crop_path and verified_label'}), 400

        # Use validation handler to handle verify classification
        success, message = validation_handler.verify_classification(
            crop_path=crop_path,
            verified_label=verified_label,
            username=session.get('username'),
            group=session.get('group')
        )
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def draw_bounding_boxes_with_numbers(image, detections):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    all_boxes = []
    for detection in detections:
        box = detection['box']
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
        all_boxes.append((left, right, top, bottom))
    for idx, detection in enumerate(detections, 1):
        box = detection['box']
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
        current_box = (left, right, top, bottom)
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        half_width = (right - left) / 2.5
        half_height = (bottom - top) / 2.5
        offset_x = half_width / 3
        offset_y = half_height / 3
        circle_radius = 3
        points = [
            (center_x, center_y),
            (center_x - offset_x, center_y),
            (center_x + offset_x, center_y),
            (center_x, center_y - offset_y),
            (center_x, center_y + offset_y)
        ]
        for point_x, point_y in points:
            point_color = "lime"
            outline_color = "darkgreen"
            for other_idx, other_box in enumerate(all_boxes):
                if other_idx != idx - 1:
                    if point_in_box(point_x, point_y, other_box):
                        point_color = "red"
                        outline_color = "darkred"
                        break
            draw.ellipse([
                (point_x - circle_radius, point_y - circle_radius),
                (point_x + circle_radius, point_y + circle_radius)
            ], fill=point_color, outline=outline_color, width=2)
        label = f"#{idx} {detection['class_name']}: {detection['score']:.2f}"
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(label) * 6, 11
        text_x = left
        text_y = top - text_height - 5
        if text_y < 0:
            text_y = bottom + 5
        draw.rectangle([
            (text_x - 2, text_y - 2),
            (text_x + text_width + 2, text_y + text_height + 2)
        ], fill="red")
        if font:
            draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            draw.text((text_x, text_y), label, fill="white")
        number_label = str(idx)
        if font:
            num_bbox = draw.textbbox((0, 0), number_label, font=font)
            num_width = num_bbox[2] - num_bbox[0]
            num_height = num_bbox[3] - num_bbox[1]
        else:
            num_width, num_height = len(number_label) * 6, 11
        draw.rectangle([
            (center_x - num_width/2 - 3, center_y - num_height/2 - 2),
            (center_x + num_width/2 + 3, center_y + num_height/2 + 2)
        ], fill="blue")
        if font:
            draw.text((center_x - num_width/2, center_y - num_height/2),
                     number_label, fill="white", font=font)
        else:
            draw.text((center_x - num_width/2, center_y - num_height/2),
                     number_label, fill="white")
    return image
    # unreachable line removed

def generate_base_filename():
    """Generate a base filename without extension and place number
    Format: YYYYMMDDHHMMSSxxxxxx (xxxxxx is random hex)"""
    timestamp = now_local().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices('0123456789abcdef', k=6))
    return f"{timestamp}{random_str}"

# Path model
DET_MODEL_PATH = os.path.join('models', 'model_automl.tflite')
CLS_MODEL_PATH = os.path.join('models', 'Mutu_320.tflite')

# =============================
# Load Model Deteksi & Klasifikasi
# =============================
det_interpreter = tf.lite.Interpreter(model_path=DET_MODEL_PATH)
det_interpreter.allocate_tensors()
det_input_details = det_interpreter.get_input_details()
det_output_details = det_interpreter.get_output_details()

cls_interpreter = tf.lite.Interpreter(model_path=CLS_MODEL_PATH)
cls_interpreter.allocate_tensors()
cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()
# Routes untuk endpoints web
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/setmutu/images/<path:filename>')
def setmutu_serve_image(filename):
    return serve_image(filename)


det_class_names = ['buah', 'batang']
cls_labels = ["ripe", "rotten", "unripe"]

def preprocess_image_det(img_path, input_size):
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(input_size)
    img_array = np.array(img_resized)

    input_dtype = det_input_details[0]['dtype']
    if input_dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img

def compute_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def non_max_suppression(detections, iou_threshold=0.6):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    selected = []
    while detections:
        current = detections.pop(0)
        selected.append(current)
        detections = [
            d for d in detections
            if compute_iou(current['box'], d['box']) <= iou_threshold
        ]
    return selected

def point_in_box(point_x, point_y, box_coords):
    left, right, top, bottom = box_coords
    return left <= point_x <= right and top <= point_y <= bottom

def boxes_overlap(box1, box2, width, height):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    left1, right1, top1, bottom1 = xmin1 * width, xmax1 * width, ymin1 * height, ymax1 * height
    left2, right2, top2, bottom2 = xmin2 * width, xmax2 * width, ymin2 * height, ymax2 * height
    return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)

def count_overlapping_points(detection, all_detections, width, height, current_idx):
    box = detection['box']
    ymin, xmin, ymax, xmax = box
    left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    half_width = (right - left) / 2
    half_height = (bottom - top) / 2
    offset_x = half_width / 3
    offset_y = half_height / 3
    points = [
        (center_x, center_y),
        (center_x - offset_x, center_y),
        (center_x + offset_x, center_y),
        (center_x, center_y - offset_y),
        (center_x, center_y + offset_y)
    ]
    overlap_count = 0
    for point_x, point_y in points:
        for other_idx, other_detection in enumerate(all_detections):
            if other_idx != current_idx:
                other_box = other_detection['box']
                other_ymin, other_xmin, other_ymax, other_xmax = other_box
                other_left = other_xmin * width
                other_right = other_xmax * width
                other_top = other_ymin * height
                other_bottom = other_ymax * height
                other_coords = (other_left, other_right, other_top, other_bottom)
                if point_in_box(point_x, point_y, other_coords):
                    overlap_count += 1
                    break
    return overlap_count

def get_overlapping_trunk(fruit_detection, trunk_detections, width, height):
    fruit_box = fruit_detection['box']
    for trunk in trunk_detections:
        trunk_box = trunk['box']
        if boxes_overlap(fruit_box, trunk_box, width, height):
            return trunk
    return None

def filter_overlapping_detections_with_trunk_validation(detections, width, height):
    """
    Proses filtering dengan alur:
    1. Deteksi overlapping (>4 titik overlap)
    2. Validasi trunk HANYA untuk buah yang overlapping
    3. Dari semua overlapping yang punya trunk, pilih score tertinggi secara global
    4. Buang overlapping yang tidak ada trunk
    5. Pertahankan semua buah non-overlapping (dengan/tanpa trunk)
    """
    fruit_detections = [d for d in detections if d['class_name'] == 'buah']
    trunk_detections = [d for d in detections if d['class_name'] == 'batang']

    # Step 1: Identifikasi buah yang overlapping (>4 titik)
    overlapping_fruits = []
    non_overlapping_fruits = []
    for idx, fruit in enumerate(fruit_detections):
        overlap_count = count_overlapping_points(fruit, fruit_detections, width, height, idx)
        if overlap_count > 4:
            overlapping_fruits.append({
                'index': idx,
                'fruit': fruit,
                'overlap_count': overlap_count
            })
        else:
            non_overlapping_fruits.append(fruit)

    validated_fruits = []
    overlapping_with_trunk = []
    overlapping_no_trunk = []
    for item in overlapping_fruits:
        fruit = item['fruit']
        trunk = get_overlapping_trunk(fruit, trunk_detections, width, height)
        if trunk:
            item['trunk'] = trunk
            overlapping_with_trunk.append(item)
        else:
            overlapping_no_trunk.append(item)

    # Step 3: Dari semua overlapping yang punya trunk, pilih satu dengan score tertinggi
    if overlapping_with_trunk:
        overlapping_with_trunk.sort(key=lambda x: x['fruit']['score'], reverse=True)
        selected_overlapping = overlapping_with_trunk[0]
        validated_fruits.append(selected_overlapping['fruit'])

    # Step 4: Buang semua overlapping fruits yang tidak ada trunk
    # (sudah otomatis tidak dimasukkan ke validated_fruits)

    # Step 5: Pertahankan SEMUA buah non-overlapping (tidak peduli ada trunk atau tidak)
    validated_fruits.extend(non_overlapping_fruits)

    # Hanya return buah saja, batang tidak diikutkan ke output akhir
    return validated_fruits

def predict_with_tflite_buah_only(image, score_threshold=0.35):
    input_shape = det_input_details[0]['shape']
    img_resized = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img_resized)
    input_dtype = det_input_details[0]['dtype']
    if input_dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    det_interpreter.set_tensor(det_input_details[0]['index'], img_array)
    det_interpreter.invoke()
    output_boxes = det_interpreter.get_tensor(det_output_details[0]['index'])[0]
    output_classes = det_interpreter.get_tensor(det_output_details[1]['index'])[0]
    output_scores = det_interpreter.get_tensor(det_output_details[2]['index'])[0]
    num_detections = int(det_interpreter.get_tensor(det_output_details[3]['index'])[0])
    detected_objects = []
    for i in range(num_detections):
        score = output_scores[i]
        if score > score_threshold:
            box = output_boxes[i]
            class_id = int(output_classes[i])
            class_name = det_class_names[class_id] if class_id < len(det_class_names) else f"Unknown_{class_id}"
            if class_name == 'buah':
                detected_objects.append({
                    'box': box,
                    'class_id': class_id,
                    'score': score,
                    'class_name': class_name
                })
    return detected_objects, image
def filter_overlapping_detections(detections, width, height):
    filtered_detections = detections.copy()
    removed_indices = set()
    while True:
        detection_to_remove = None
        max_overlap = 0
        for idx, detection in enumerate(filtered_detections):
            if idx in removed_indices:
                continue
            overlap_count = count_overlapping_points_excluding_removed(
                detection, filtered_detections, width, height, idx, removed_indices
            )
            if overlap_count > 4 and overlap_count > max_overlap:
                max_overlap = overlap_count
                detection_to_remove = idx
        if detection_to_remove is None:
            break
        removed_indices.add(detection_to_remove)
    for idx in sorted(removed_indices, reverse=True):
        filtered_detections.pop(idx)
    return filtered_detections

def count_overlapping_points_excluding_removed(detection, all_detections, width, height, current_idx, removed_indices):
    box = detection['box']
    ymin, xmin, ymax, xmax = box
    left, right, top, bottom = xmin * width, xmax * width, ymin * height, ymax * height
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    half_width = (right - left) / 2
    half_height = (bottom - top) / 2
    offset_x = half_width / 3
    offset_y = half_height / 3
    points = [
        (center_x, center_y),
        (center_x - offset_x, center_y),
        (center_x + offset_x, center_y),
        (center_x, center_y - offset_y),
        (center_x, center_y + offset_y)
    ]
    overlap_count = 0
    for point_x, point_y in points:
        for other_idx, other_detection in enumerate(all_detections):
            if other_idx != current_idx and other_idx not in removed_indices:
                other_box = other_detection['box']
                other_ymin, other_xmin, other_ymax, other_xmax = other_box
                other_left = other_xmin * width
                other_right = other_xmax * width
                other_top = other_ymin * height
                other_bottom = other_ymax * height
                other_coords = (other_left, other_right, other_top, other_bottom)
                if point_in_box(point_x, point_y, other_coords):
                    overlap_count += 1
                    break
    return overlap_count

def crop_detections(image, detections, target_class='buah', target_size=(896, 896)):
    width, height = image.size
    cropped_images = []
    for det in detections:
        if det['class_name'] != target_class:
            continue
        ymin, xmin, ymax, xmax = det['box']
        left, top, right, bottom = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

        padding = 2
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)

        cropped_img = image.crop((left, top, right, bottom))
        cropped_img = cropped_img.resize(target_size, Image.LANCZOS)

        enhancer = ImageEnhance.Sharpness(cropped_img)
        cropped_img = enhancer.enhance(1.5)

        cropped_images.append((cropped_img, (left, top, right, bottom)))
    return cropped_images

def preprocess_image_cls(pil_img, input_size):
    img_resized = pil_img.resize(input_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def _normalize_label(label: str) -> str:
    """Map any legacy/variant labels to 3-class set."""
    if not label:
        return 'unripe'
    l = str(label).strip()
    # accept either lowercase or capitalized legacy values
    l_low = l.lower()
    if l_low in ('ripe',):
        return 'ripe'
    if l_low in ('rotten', 'overripe'):
        return 'rotten'
    if l_low in ('unripe',):
        return 'unripe'
    # map anything else to closest bucket (default to unripe)
    return 'unripe'

def classify_crops(cropped_images):
    results = []
    input_size = (cls_input_details[0]['shape'][1], cls_input_details[0]['shape'][2])

    for crop, bbox in cropped_images:
        input_data = preprocess_image_cls(crop, input_size)
        cls_interpreter.set_tensor(cls_input_details[0]['index'], input_data)
        cls_interpreter.invoke()

        output_data = cls_interpreter.get_tensor(cls_output_details[0]['index'])[0]
        pred_class = np.argmax(output_data)
        pred_conf = output_data[pred_class]

        results.append({
            'crop': crop,
            'bbox': bbox,
            'label': _normalize_label(cls_labels[pred_class]),
            'confidence': float(pred_conf)
        })
    return results

@app.route('/train')
def train():
    """Training page with testing and gallery features"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/traincolab')
def traincolab():
    """Redirect to Google Colab for training"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('traincolab.html')

@app.route('/classify_single', methods=['POST'])
def classify_single():
    """Classify a single image without detection"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read and preprocess image
        image_stream = file.read()
        image = Image.open(io.BytesIO(image_stream)).convert('RGB')
        
        # Resize to classification model input size
        input_size = (cls_input_details[0]['shape'][1], cls_input_details[0]['shape'][2])
        img_resized = image.resize(input_size)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run classification
        cls_interpreter.set_tensor(cls_input_details[0]['index'], img_array)
        cls_interpreter.invoke()
        output_data = cls_interpreter.get_tensor(cls_output_details[0]['index'])[0]
        
        # Get predictions
        pred_class = np.argmax(output_data)
        pred_conf = float(output_data[pred_class])
        pred_label = _normalize_label(cls_labels[pred_class])
        
        # Get all probabilities
        probabilities = {
            'ripe': float(output_data[0]),
            'rotten': float(output_data[1]),
            'unripe': float(output_data[2])
        }
        
        return jsonify({
            'label': pred_label,
            'confidence': pred_conf,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_images', methods=['POST'])
def get_training_images():
    """Get images from training dataset folder (deprecated - use get_dataset_images)"""
    return get_dataset_images()

@app.route('/get_dataset_images', methods=['POST'])
def get_dataset_images():
    """Get images from static/dataset folder"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        class_name = data.get('class_name', 'ripe')
        page = data.get('page', 1)
        per_page = data.get('per_page', 50)
        
        # Always use static/dataset path
        dataset_path = Path('static') / 'dataset'
        
        # Get class folder path (capitalize first letter)
        class_folder = dataset_path / class_name.capitalize()
        
        if not class_folder.exists():
            return jsonify({'error': f'Dataset folder not found: {class_folder}', 'images': []})
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        all_images = []
        
        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                # Use relative path from static folder for web serving
                relative_path = f"static/dataset/{class_name.capitalize()}/{img_file.name}"
                all_images.append({
                    'path': relative_path,
                    'name': img_file.name,
                    'class': class_name
                })
        
        # Sort by name
        all_images.sort(key=lambda x: x['name'])
        
        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_images = all_images[start_idx:end_idx]
        
        return jsonify({
            'images': paginated_images,
            'total': len(all_images),
            'page': page,
            'per_page': per_page,
            'has_more': end_idx < len(all_images)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_folder', methods=['POST'])
def classify_folder():
    """Classify multiple images from a folder"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        files = request.files.getlist('files')
        folder_name = request.form.get('folder_name', 'Unknown Folder')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Classification results
        results = {
            'ripe': 0,
            'rotten': 0,
            'unripe': 0
        }
        
        input_size = (cls_input_details[0]['shape'][1], cls_input_details[0]['shape'][2])
        
        # Process each file
        for file in files:
            try:
                # Read image
                image_stream = file.read()
                image = Image.open(io.BytesIO(image_stream)).convert('RGB')
                
                # Resize and preprocess
                img_resized = image.resize(input_size)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Run classification
                cls_interpreter.set_tensor(cls_input_details[0]['index'], img_array)
                cls_interpreter.invoke()
                output_data = cls_interpreter.get_tensor(cls_output_details[0]['index'])[0]
                
                # Get prediction
                pred_class = np.argmax(output_data)
                pred_label = _normalize_label(cls_labels[pred_class])
                
                # Update results
                results[pred_label] += 1
                
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        total = sum(results.values())
        
        if total == 0:
            return jsonify({'error': 'No images could be classified'}), 400
        
        return jsonify({
            'folder_name': folder_name,
            'total': total,
            'summary': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    import time
    
    # Start timing
    start_time = time.time()
    
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image_stream = file.read()
        image = Image.open(io.BytesIO(image_stream)).convert('RGB')
        width, height = image.size
        
        # Start detection timing
        detection_start = time.time()
        
        # 1. Deteksi buah & batang
        input_shape = det_input_details[0]['shape']
        img_resized = image.resize((input_shape[1], input_shape[2]))
        img_array = np.array(img_resized)
        input_dtype = det_input_details[0]['dtype']
        if input_dtype == np.uint8:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        det_interpreter.set_tensor(det_input_details[0]['index'], img_array)
        det_interpreter.invoke()
        output_boxes = det_interpreter.get_tensor(det_output_details[0]['index'])[0]
        output_classes = det_interpreter.get_tensor(det_output_details[1]['index'])[0]
        output_scores = det_interpreter.get_tensor(det_output_details[2]['index'])[0]
        num_detections = int(det_interpreter.get_tensor(det_output_details[3]['index'])[0])
        
        # End detection timing
        detection_end = time.time()
        detection_time = detection_end - detection_start
        detections = []
        for i in range(num_detections):
            score = float(output_scores[i])
            class_id = int(output_classes[i])
            class_name = det_class_names[class_id] if class_id < len(det_class_names) else f"Unknown_{class_id}"
            box = output_boxes[i]
            # Threshold sesuai ObjectDetectionPy
            if class_name == 'buah' and score >= 0.3618514:
                detections.append({'box': box, 'class_id': class_id, 'score': score, 'class_name': class_name})
            elif class_name == 'batang' and score >= 0.3618514:
                detections.append({'box': box, 'class_id': class_id, 'score': score, 'class_name': class_name})

        # 2. NMS khusus untuk buah
        def iou_special(box1, box2):
            ymin1, xmin1, ymax1, xmax1 = box1
            ymin2, xmin2, ymax2, xmax2 = box2
            x1 = max(xmin1, xmin2)
            y1 = max(ymin1, ymin2)
            x2 = min(xmax1, xmax2)
            y2 = min(ymax1, ymax2)
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            intersection = inter_w * inter_h
            area1 = max(0.0, (xmax1 - xmin1) * (ymax1 - ymin1))
            if area1 <= 0:
                return 0.0
            return (intersection / area1) if intersection > 0 else 0.0

        def nms_bunch(dets, threshold=0.5):
            dets = [d for d in dets if d['class_name'] == 'buah']
            dets = sorted(dets, key=lambda d: d['score'], reverse=True)
            n = len(dets)
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                box1 = dets[i]['box']
                score1 = dets[i]['score']
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    box2 = dets[j]['box']
                    score2 = dets[j]['score']
                    iou = iou_special(box1, box2)
                    if iou >= threshold:
                        if score1 >= score2:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break
            return [dets[i] for i in range(n) if keep[i]]

        # 3. Rule-based counting (clustering) jika submission > 20
        submission = int(request.form.get('submission', 25)) if request.form.get('submission') else 25
        bunch_detections = [d for d in detections if d['class_name'] == 'buah']
        trunk_detections = [d for d in detections if d['class_name'] == 'batang']
        if submission > 20:
            bunch_nms = nms_bunch(bunch_detections)
            # Dummy clustering: totalBunch = min(submission, len(bunch_nms))
            # (implementasi clustering penuh bisa ditambah jika diperlukan)
            final_bunch = min(submission, len(bunch_nms))
            detection_count = final_bunch
            filtered_detections = bunch_nms[:final_bunch]
        else:
            filtered_detections = bunch_detections
            detection_count = len(filtered_detections)

        # 4. Visualisasi bounding box dan nomor
        detection_image = image.copy()
        detection_image = draw_bounding_boxes_with_numbers(detection_image, filtered_detections)

        # 5. Cropping dan klasifikasi
        # Start classification timing
        classification_start = time.time()
        
        cropped_images = crop_detections(image, filtered_detections)
        results = classify_crops(cropped_images)
        
        # End classification timing
        classification_end = time.time()
        classification_time = classification_end - classification_start
        
        classification_summary = Counter()
        crops_info = []

        # Store original image in memory
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='JPEG')
        original_img_str = base64.b64encode(original_buffer.getvalue()).decode()

        # Store detection image in memory
        detection_buffer = io.BytesIO()
        detection_image.save(detection_buffer, format='JPEG')
        detection_img_str = base64.b64encode(detection_buffer.getvalue()).decode()
        
        # Process all crops
        for i, result in enumerate(results, 1):
            # Update summary
            classification_summary[result['label']] += 1
            
            # Save crop to buffer for display
            buffer = io.BytesIO()
            result['crop'].save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            crops_info.append({
                'image_data': f'data:image/jpeg;base64,{img_str}',
                'label': result['label'],
                'confidence': result['confidence'],
                'bbox': result['bbox']
            })
        # normalize classification_summary to only target labels
        # ensure all three keys exist
        normalized_summary = {k: 0 for k in cls_labels}
        for label, cnt in classification_summary.items():
            normalized_summary[_normalize_label(label)] += cnt

        # Calculate total process time
        end_time = time.time()
        total_process_time = end_time - start_time

        return jsonify({
            'original_image': f"data:image/jpeg;base64,{original_img_str}",
            'detection_image': f"data:image/jpeg;base64,{detection_img_str}",
            'detection_count': detection_count,
            'classification_summary': normalized_summary,
            'crops': crops_info,
            'original_filename': file.filename,
            'timing_info': {
                'total_process_time': round(total_process_time, 3),
                'detection_time': round(detection_time, 3),
                'classification_time': round(classification_time, 3)
            }
        })




# Route for saving individual crops
@app.route('/save_crop', methods=['POST'])
def save_crop_endpoint():
    try:
        data = request.get_json()
        crop_path = data.get('crop_path')
        class_name = data.get('class_name')

        # Validate class name
        class_name = _normalize_label(class_name)
        if class_name not in CLASS_LABELS:
            return jsonify({'success': False, 'error': f'Invalid class name: {class_name}'})

        # Extract the base64 image data
        if crop_path.startswith('data:image'):
            # Split the base64 string at ',' to get the actual data
            base64_data = crop_path.split(',')[1]
            # Convert base64 to image
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
        else:
            # If it's a file path, load the image directly from configured storage
            image = Image.open(_image_fs_path(crop_path))

        # Generate new base filename
        base_name = generate_base_filename()
        # Since new base generated, start place numbering at 1
        place = 1
        # Create filename with place number
        filename = f"{base_name}_{place}.jpg"
        save_path = _image_fs_path(filename)

        # Save the image
        image.save(save_path)
        
        # Add to database
        _add_ticket_record(filename, class_name, place)

        return jsonify({
            'success': True,
            'saved_path': _image_url(filename)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Route for saving all classifications
@app.route('/save_all_classifications', methods=['POST'])
def save_all_classifications():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract timing information from the request
        timing_info = data.get('timing_info', {})
        total_process_time = timing_info.get('total_process_time', 0.0)
        detection_time = timing_info.get('detection_time', 0.0)
        classification_time = timing_info.get('classification_time', 0.0)

        # Generate base filename using the existing function - SATU base_name untuk semua file
        base_name = generate_base_filename()
        extension = '.jpg'
        print(f"Generated base_name: {base_name}")  # Debug log
        print(f"Timing info - Total: {total_process_time}s, Detection: {detection_time}s, Classification: {classification_time}s")
        
        saved_files = []  # Track semua file yang disimpan
        
        # 1. Save original image with (-) place number
        original_image_data = data.get('original_image')
        if original_image_data:
            if ',' in original_image_data:
                image_data = original_image_data.split(',')[1]
            else:
                image_data = original_image_data
            image_bytes = base64.b64decode(image_data)
            
            # Save original image with (-) place number
            original_filename = f"{base_name}(-){extension}"
            original_path = _image_fs_path(original_filename)
            
            with open(original_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Saved original: {original_filename}")  # Debug log
            saved_files.append(original_filename)
            _add_ticket_record(original_filename, "original", "-", False, 
                             total_process_time, detection_time, classification_time)
        
        # 2. Save detection image with (0)
        detection_image_data = data.get('detection_image')
        if detection_image_data:
            if ',' in detection_image_data:
                image_data = detection_image_data.split(',')[1]
            else:
                image_data = detection_image_data
            image_bytes = base64.b64decode(image_data)
            
            # Save detection image with (0)
            detection_filename = f"{base_name}(0){extension}"
            detection_path = _image_fs_path(detection_filename)
            
            with open(detection_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Saved detection: {detection_filename}")  # Debug log
            saved_files.append(detection_filename)
            _add_ticket_record(detection_filename, "detection", 0, False,
                             total_process_time, detection_time, classification_time)

        # 3. Save each crop with sequential numbers in parentheses
        crops_data = data.get('crops', [])
        print(f"Processing {len(crops_data)} crops")
        
        for i, crop in enumerate(crops_data, 1):
            if not crop.get('image_data'):
                print(f"Skipping crop {i}: No image data")
                continue
                
            # Extract base64 data - remove header if present
            image_data = crop['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                print(f"Error decoding crop {i}: {str(e)}")
                continue
            
            # Save crop with sequential number in parentheses - USING SAME base_name
            crop_filename = f"{base_name}({i}){extension}"
            crop_path = _image_fs_path(crop_filename)
            
            try:
                with open(crop_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Add to database with the classification label and place number
                classification_label = _normalize_label(crop.get('label', 'unripe'))
                _add_ticket_record(crop_filename, classification_label, i, False,
                                 total_process_time, detection_time, classification_time)
                
                print(f"Saved crop {i}: {crop_filename} with label {classification_label}")
                saved_files.append(crop_filename)
            except Exception as e:
                print(f"Error saving crop {i}: {str(e)}")
                continue

        # Return success only if we saved at least original, detection, and one crop
        if len(saved_files) >= 3:
            print(f"Successfully saved {len(saved_files)} files with base_name: {base_name}")
            return jsonify({
                'success': True,
                'message': f'Saved {len(saved_files)} files successfully',
                'saved_files': saved_files,
                'base_filename': base_name
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Only saved {len(saved_files)} files, expected at least 3 (original, detection, and crops)',
                'saved_files': saved_files
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route for checking database status
@app.route('/database_status')
def database_status():
    database = validation_handler.load_database()
    class_counts = Counter(_normalize_label(entry['class_result']) for entry in database if isinstance(entry['place'], (int, float)) and entry['place'] > 0)
    total_images = sum(class_counts.values())
    
    return jsonify({
        'class_counts': dict(class_counts),
        'total_images': total_images
    })

# Route for getting timing statistics
@app.route('/timing_statistics')
def timing_statistics():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        group_id = request.args.get('group_id')
        date_filter = request.args.get('date')
        
        stats = validation_handler.get_timing_statistics(group_id, date_filter)
        
        if stats:
            return jsonify({
                'success': True,
                'statistics': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No timing data found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# History Routes - Separate from main functionality
@app.route('/history')
def history():
    """Route for history page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('history.html')

@app.route('/ticket')
def ticket():
    """Route for ticket page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('ticket.html')

@app.route('/get_validate_history_by_date', methods=['POST'])
def get_validate_history_by_date():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    """Get validation history for a specific date from validations table"""
    try:
        data = request.get_json()
        selected_date = data.get('date')
        if not selected_date:
            return jsonify({'error': 'No date provided'}), 400

        # Get current user's group from session
        current_user_group = session.get('group', 'user')
        # Load validations from DB
        validate_data = validation_handler.load_validations()
        
        # Filter by user's group and date, then group by (username_yang_mengubah, base_name, timestamp)
        grouped_items = {}

        for entry in validate_data:
            # Required fields from validate.txt
            if not isinstance(entry, dict):
                continue
            if str(entry.get('group')) != str(current_user_group):
                continue

            ts_str = str(entry.get('timestamp', ''))
            try:
                entry_date = ts_str.split(' ')[0]
                if entry_date != selected_date:
                    continue
            except Exception:
                continue

            filename = str(entry['file_name'])
            base_name = filename.split('(')[0] if '(' in filename else filename.rsplit('.', 1)[0]
            username_change = entry.get('username_yang_mengubah', 'Unknown')
            # use full timestamp string as batch id
            batch_ts = ts_str

            key = f"{base_name}|{username_change}|{batch_ts}"
            gi = grouped_items.setdefault(key, {
                'base_name': base_name,
                'username_yang_mengubah': username_change,
                'batch_timestamp': batch_ts,
                'entries': []
            })
            gi['entries'].append(entry)

        # Build list
        result_items = []
        for key, gi in grouped_items.items():
            entries = gi['entries']
            classifications = {}
            valid_count = 0
            invalid_count = 0
            for e in entries:
                cls_name = e.get('class_result', 'Unknown')
                classifications[cls_name] = classifications.get(cls_name, 0) + 1
                if bool(e.get('valid_status')):
                    valid_count += 1
                else:
                    invalid_count += 1

            # status: if any invalid then open, else close
            status = 'open' if invalid_count > 0 else 'close'

            # For display time, take time part
            time_part = gi['batch_timestamp'].split(' ')[1] if ' ' in gi['batch_timestamp'] else gi['batch_timestamp']

            result_items.append({
                'id': key,
                'filename': f"{gi['base_name']}(-).jpg",
                'image_path': _image_url(f"{gi['base_name']}(-).jpg"),
                'timestamp': time_part,
                'detection_count': len(entries),
                'classifications': classifications,
                'status': status,
                'uploader': gi['username_yang_mengubah'],
                'can_edit': True,
                'valid_count': valid_count,
                'invalid_count': invalid_count
            })

        # Sort by full batch timestamp desc (string time within the day)
        result_items.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'items': result_items})

    except Exception as e:
        print(f"Error in get_validate_history_by_date: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_validate_history_detail', methods=['POST'])
def get_validate_history_detail():
    """Return details for a grouped validate history id built as base|username|timestamp"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    try:
        data = request.get_json() or {}
        group_id = data.get('id')
        if not group_id or '|' not in group_id:
            return jsonify({'error': 'Invalid id'}), 400

        base_name, username_change, batch_ts = group_id.split('|', 2)
        current_user_group = session.get('group', 'user')

        # Load validations from DB
        validate_data = validation_handler.load_validations()

        # Filter entries for this group id
        items = []
        for e in validate_data:
            if not isinstance(e, dict):
                continue
            if str(e.get('group')) != str(current_user_group):
                continue
            if e.get('username_yang_mengubah') != username_change:
                continue
            if str(e.get('timestamp')) != str(batch_ts):
                continue
            fname = str(e.get('file_name', ''))
            if not fname.startswith(base_name):
                continue
            items.append({
                'file_name': fname,
                'image_path': _image_url(fname),
                'class_result': e.get('class_result', 'Unknown'),
                'valid_status': bool(e.get('valid_status', False)),
                'place': e.get('place')
            })

        # Sort by place if numeric in filename
        def place_key(it):
            try:
                # expect like base(5).jpg
                inside = it['file_name'].split('(')[1].split(')')[0]
                return int(inside)
            except Exception:
                return 0
        items.sort(key=place_key)

        # Build summary
        summary = {}
        for it in items:
            cls_name = it['class_result']
            summary[cls_name] = summary.get(cls_name, 0) + 1

        return jsonify({
            'base_image': _image_url(f"{base_name}(-).jpg"),
            'timestamp': batch_ts,
            'username_yang_mengubah': username_change,
            'items': items,
            'summary': summary,
            'count': len(items)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_history_by_date', methods=['POST'])
def get_history_by_date():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    """Get historical detections for a specific date"""
    try:
        data = request.get_json()
        selected_date = data.get('date')
        if not selected_date:
            return jsonify({'error': 'No date provided'}), 400

        # Get current user's group from session
        current_user_group = session.get('group', 'user')
        
        database = validation_handler.load_database()
        original_images = []

        # Filter only original images (place="-") for the selected date and user's group
        for entry in database:
            if (entry['place'] == '-' and 
                entry.get('group') == current_user_group):  # Filter by user's group
                filename = entry['file_name']
                # Extract date from filename (assuming format: YYYYMMDDHHMMSS...)
                file_date = filename[:8]  # Get YYYYMMDD part
                formatted_date = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:]}"

                if formatted_date == selected_date:
                    # Get timestamp for display
                    timestamp = filename[8:14]  # Get HHMMSS part
                    formatted_time = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}"

                    # Get base name without extension for later use
                    base_name = filename.split('(')[0]

                    # Count total detections for this image (only from same group)
                    detection_count = sum(1 for d in database 
                                        if (d['file_name'].startswith(base_name) and 
                                            d['place'] != '-' and d['place'] != 0 and
                                            d.get('group') == current_user_group))

                    # Get classification summary (only from same group)
                    classifications = {}
                    for d in database:
                        if (d['file_name'].startswith(base_name) and 
                            d['place'] not in ['-', 0] and
                            d.get('group') == current_user_group):
                            class_name = _normalize_label(d['class_result'])
                            if class_name not in classifications:
                                classifications[class_name] = 0
                            classifications[class_name] += 1

                    # Determine status: if any entry has status 'close' -> close, else 'open'
                    statuses = [str(d.get('status', 'open')).lower() for d in database if (d['file_name'].startswith(base_name) and d.get('group') == current_user_group)]
                    if any(s == 'close' or s == 'true' for s in statuses):
                        status = 'close'
                    else:
                        status = 'open'

                    uploader_username = entry.get('username')

                    original_images.append({
                        'id': base_name,
                        'filename': filename,
                        'image_path': _image_url(filename),
                        'timestamp': formatted_time,
                        'detection_count': detection_count,
                        'classifications': classifications,
                        'status': status,
                        'uploader': uploader_username,
                        'can_edit': uploader_username == session.get('username')
                    })
        
        # Sort by timestamp in descending order (newest first)
        original_images.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'items': original_images})

        # Group entries by base filename
        for entry in database:
            filename = entry['file_name']
            base_name = filename.split('(')[0] if '(' in filename else filename.split('.')[0]
            
            # Extract timestamp from base_name (assuming it's in the format YYYYMMDD_HHMMSS_...)
            try:
                timestamp_str = base_name.split('_')[0] + '_' + base_name.split('_')[1]
                entry_date = datetime.strptime(timestamp_str[:8], '%Y%m%d').strftime('%Y-%m-%d')
                
                if entry_date == selected_date:
                    if base_name not in date_items:
                        date_items[base_name] = {
                            'id': base_name,
                            'timestamp': datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%H:%M:%S'),
                            'detection_count': 0,
                            'crops': [],
                            'image_path': None
                        }
                    
                    if entry['place'] == '-':
                        # Original image
                        date_items[base_name]['image_path'] = _image_url(filename)
                    elif entry['place'] > 0:
                        # Count detections
                        date_items[base_name]['detection_count'] += 1
                        date_items[base_name]['crops'].append({
                            'image': _image_url(filename),
                            'classification': entry['class_result'],
                            'confidence': 100  # Add actual confidence if available
                        })
            except (ValueError, IndexError):
                # Skip entries with invalid timestamp format
                continue

        # Convert to list and sort by timestamp
        items = list(date_items.values())
        items.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'items': [item for item in items if item['image_path']]  # Only return items with original images
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# History detail route - Separate from main detection functionality
@app.route('/historydetail')
def history_detail():
    """Render the history detail page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('historydetail.html')

@app.route('/ticketsession')
def ticket_session():
    """Render the ticket session page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('ticketsession.html')

@app.route('/validate')
def validate_page():
    """Render the validation page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('validate.html')

@app.route('/get_monthly_stats', methods=['POST'])
def get_monthly_stats():
    """Get monthly statistics for ticket dashboard"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        year = data.get('year')
        month = data.get('month')  # 0-based month (0 = January)
        
        if year is None or month is None:
            return jsonify({'error': 'Year and month are required'}), 400
            
        # Get current user's group from session
        current_user_group = session.get('group', 'user')
        
        # Get monthly ticket count from database
        count = validation_handler.get_monthly_ticket_count(year, month + 1, current_user_group)  # Convert to 1-based month
        
        return jsonify({
            'success': True,
            'count': count,
            'year': year,
            'month': month
        })
        
    except Exception as e:
        print(f"Error in get_monthly_stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_detection_details', methods=['POST'])
def get_detection_details():
    """Get historical detection details for viewing"""
    try:
        data = request.get_json()
        base_name = data.get('id')
        if not base_name:
            return jsonify({'error': 'No ID provided'}), 400

        # Get current user's group from session
        current_user_group = session.get('group', 'user')

        database = validation_handler.load_database()
        details = {
            'original_image': None,
            'detection_image': None,
            'crops': [],
            'summary': {}
        }

        # Get all entries for this image set from the same group
        entries = [entry for entry in database 
                  if (entry['file_name'].startswith(base_name) and
                      entry.get('group') == current_user_group)]
        
        # Calculate classification summary
        class_summary = {}
        for entry in entries:
            if entry['place'] not in ['-', 0]:
                class_name = _normalize_label(entry['class_result'])
                if class_name not in class_summary:
                    class_summary[class_name] = 0
                class_summary[class_name] += 1

        for entry in entries:
            filename = entry['file_name']
            image_path = _image_url(filename)

            if entry['place'] == '-':
                details['original_image'] = image_path
            elif entry['place'] == 0:
                details['detection_image'] = image_path
            elif isinstance(entry['place'], (int, float)) and entry['place'] > 0:
                details['crops'].append({
                    'image': image_path,
                    'classification': entry['class_result'],
                    'place': entry['place']
                })

        # Sort crops by place number
        details['crops'].sort(key=lambda x: x['place'])
        details['summary'] = class_summary
        details['total_detections'] = len(details['crops'])
        
        # Get timestamp from base_name
        timestamp = base_name[8:14]  # HHMMSS part
        details['timestamp'] = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}"

        return jsonify(details)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Routes with /setmutu/ prefix for UI navigation
@app.route('/setmutu/')
def setmutu_index():
    return redirect('/index')

@app.route('/setmutu/index')
def setmutu_dashboard():
    return dashboard()

@app.route('/setmutu/ticket')
def setmutu_ticket():
    return ticket()

@app.route('/setmutu/history')
def setmutu_history():
    return history()

@app.route('/setmutu/train')
def setmutu_train():
    return train()

@app.route('/setmutu/traincolab')
def setmutu_traincolab():
    return traincolab()

@app.route('/setmutu/classify_single', methods=['POST'])
def setmutu_classify_single():
    return classify_single()

@app.route('/setmutu/get_training_images', methods=['POST'])
def setmutu_get_training_images():
    return get_training_images()

@app.route('/setmutu/get_dataset_images', methods=['POST'])
def setmutu_get_dataset_images():
    return get_dataset_images()

@app.route('/setmutu/classify_folder', methods=['POST'])
def setmutu_classify_folder():
    return classify_folder()

@app.route('/setmutu/logout')
def setmutu_logout():
    return logout()

@app.route('/setmutu/validate')
def setmutu_validate():
    return validate_page()

@app.route('/setmutu/register', methods=['GET', 'POST'])
def setmutu_register():
    return register()

@app.route('/setmutu/historydetail')
def setmutu_history_detail():
    return history_detail()

@app.route('/setmutu/ticketsession')
def setmutu_ticket_session():
    return ticket_session()

@app.route('/setmutu/static/<path:filename>')
def setmutu_static(filename):
    return send_from_directory('static', filename)

@app.route('/setmutu/download/<path:filename>')
def setmutu_download(filename):
    return download_file(filename)

@app.route('/setmutu/upload', methods=['POST'])
def setmutu_upload():
    return upload_file()

@app.route('/setmutu/save_crop', methods=['POST'])
def setmutu_save_crop():
    return save_crop_endpoint()

@app.route('/setmutu/save_all_classifications', methods=['POST'])
def setmutu_save_all_classifications():
    return save_all_classifications()

@app.route('/setmutu/update_valid_status', methods=['POST'])
def setmutu_update_valid_status():
    return update_valid_status()

@app.route('/setmutu/submit_validations', methods=['POST'])
def setmutu_submit_validations():
    return submit_validations()

@app.route('/setmutu/verify', methods=['POST'])
def setmutu_verify():
    return verify_classification()

@app.route('/setmutu/database_status')
def setmutu_database_status():
    return database_status()

@app.route('/setmutu/timing_statistics')
def setmutu_timing_statistics():
    return timing_statistics()

@app.route('/setmutu/get_validate_history_by_date', methods=['POST'])
def setmutu_get_validate_history_by_date():
    return get_validate_history_by_date()

@app.route('/setmutu/get_validate_history_detail', methods=['POST'])
def setmutu_get_validate_history_detail():
    return get_validate_history_detail()

@app.route('/setmutu/get_monthly_stats', methods=['POST'])
def setmutu_get_monthly_stats():
    return get_monthly_stats()

@app.route('/setmutu/get_history_by_date', methods=['POST'])
def setmutu_get_history_by_date():
    return get_history_by_date()

@app.route('/setmutu/get_detection_details', methods=['POST'])
def setmutu_get_detection_details():
    return get_detection_details()

@app.route('/setmutu/get_monthly_statistics', methods=['POST'])
def setmutu_get_monthly_statistics():
    """Get monthly statistics for entire month regardless of date filter"""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        year = data.get('year')
        month = data.get('month')  # Optional - if not provided, get whole year
        
        if year is None:
            return jsonify({'error': 'Year is required'}), 400
            
        # Get current user's group from session
        current_user_group = session.get('group', 'user')
        
        # Get statistics from database
        if month is not None:
            # Get data for specific month
            items = validation_handler.get_monthly_ticket_data(year, month + 1, current_user_group)  # Convert to 1-based month
        else:
            # Get data for entire year
            items = validation_handler.get_yearly_ticket_data(year, current_user_group)
        
        return jsonify({
            'success': True,
            'items': items,
            'year': year,
            'month': month,
            'count': len(items)
        })
        
    except Exception as e:
        print(f"Error in setmutu_get_monthly_statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/setmutu/set_status', methods=['POST'])
def setmutu_set_status():
    return set_status()

if __name__ == '__main__':
    app.run(debug=True)