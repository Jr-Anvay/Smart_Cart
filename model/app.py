# from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
# from collections import defaultdict
# import threading
# import json
# import cv2
# import numpy as np
# import time
# import os
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load YOLO model and product data
# model = YOLO("C:/Study Material/model/my_model/train/weights/best.pt")
# with open("C:/Study Material/model/product_info.json", "r") as f:
#     product_data = json.load(f)
    
# # Global variables
# detected_items = []
# latest_capture = None

# def capture_single_image():
#     """Capture a single image from camera"""
#     try:
#         # Try different camera backends for better compatibility
#         cap = None
#         for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
#             try:
#                 if backend:
#                     cap = cv2.VideoCapture(0, backend)
#                 else:
#                     cap = cv2.VideoCapture(0)
                
#                 if cap.isOpened():
#                     # Test if we can read a frame
#                     ret, test_frame = cap.read()
#                     if ret and test_frame is not None:
#                         cap.release()
#                         break
#                 cap.release()
#             except:
#                 if cap:
#                     cap.release()
#                 continue
        
#         # Final capture
#         if backend:
#             cap = cv2.VideoCapture(0, backend)
#         else:
#             cap = cv2.VideoCapture(0)
            
#         if not cap.isOpened():
#             return None, "Could not open camera"
        
#         # Set camera properties
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Warm up camera - take a few frames
#         for i in range(5):
#             ret, frame = cap.read()
#             if ret:
#                 break
#             time.sleep(0.1)
        
#         # Capture the actual image
#         ret, frame = cap.read()
#         cap.release()
        
#         if ret and frame is not None:
#             return frame, None
#         else:
#             return None, "Failed to capture image"
            
#     except Exception as e:
#         if cap:
#             cap.release()
#         return None, f"Camera error: {str(e)}"

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/capture_image')
# def capture_image():
#     global detected_items, latest_capture
    
#     # Capture image from camera
#     frame, error = capture_single_image()
    
#     if error:
#         return jsonify({'error': error})
    
#     if frame is None:
#         return jsonify({'error': 'No frame captured'})
    
#     # Save captured image
#     timestamp = str(int(time.time()))
#     save_path = f'static/captures/capture_{timestamp}.jpg'
#     os.makedirs('static/captures', exist_ok=True)
#     cv2.imwrite(save_path, frame)
    
#     # Store latest capture for display
#     latest_capture = save_path
    
#     # Run YOLO detection
#     try:
#         results = model(frame)[0]
#         boxes = results.boxes
#         temp_cart = defaultdict(int)
        
#         if boxes is not None and len(boxes) > 0:
#             for i in range(len(boxes)):
#                 class_id = int(boxes.cls[i])
#                 name = model.names[class_id]
#                 confidence = float(boxes.conf[i])
                
#                 # Only add items with confidence > 0.5
#                 if confidence > 0.5:
#                     temp_cart[name] += 1
        
#         # Add detected items to cart
#         for product, qty in temp_cart.items():
#             price = product_data.get(product, {}).get("price", 0)
#             img = product_data.get(product, {}).get("image", "")
#             total = qty * price
            
#             # Check if item already exists in cart
#             existing_item = None
#             for item in detected_items:
#                 if item["name"] == product:
#                     existing_item = item
#                     break
            
#             if existing_item:
#                 existing_item["qty"] += qty
#                 existing_item["total"] = existing_item["qty"] * existing_item["price"]
#             else:
#                 detected_items.append({
#                     "name": product,
#                     "qty": qty,
#                     "price": price,
#                     "total": total,
#                     "image": img
#                 })
        
#         return jsonify({
#             'status': 'success', 
#             'items_detected': len(temp_cart),
#             'capture_path': save_path,
#             'capture_url': f'/{save_path}'
#         })
    
#     except Exception as e:
#         return jsonify({'error': f'Detection failed: {str(e)}'})

# @app.route('/get_latest_capture')
# def get_latest_capture():
#     global latest_capture
#     if latest_capture:
#         return jsonify({'capture_url': f'/{latest_capture}'})
#     return jsonify({'capture_url': None})

# @app.route('/cart')
# def cart():
#     return render_template('cart.html')

# @app.route('/get_cart_items')
# def get_cart_items():
#     total = sum(item["total"] for item in detected_items)
#     return jsonify({"items": detected_items, "total": total})

# @app.route('/update_cart', methods=['POST'])
# def update_cart():
#     data = request.get_json()
#     item_name = data.get('name')
#     new_qty = int(data.get('qty', 0))
    
#     for item in detected_items:
#         if item['name'] == item_name:
#             if new_qty <= 0:
#                 detected_items.remove(item)
#             else:
#                 item['qty'] = new_qty
#                 item['total'] = item['qty'] * item['price']
#             break
    
#     return jsonify({'status': 'success'})

# @app.route('/clear_cart')
# def clear_cart():
#     global detected_items
#     detected_items = []
#     return jsonify({'status': 'success'})

# @app.route('/payment')
# def payment():
#     total = sum(item["total"] for item in detected_items)
#     return render_template('payment.html', total=total)

# @app.route('/process_payment', methods=['POST'])
# def process_payment():
#     global detected_items
    
#     payment_method = request.form.get('payment_method')
#     total_amount = sum(item["total"] for item in detected_items)
    
#     # Clear cart after successful payment
#     detected_items = []
    
#     return render_template('payment_success.html', 
#                          amount=total_amount, 
#                          method=payment_method)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from collections import defaultdict
import threading
import json
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model and product data
model = YOLO("C:/Study Material/model/my_model/train/weights/best.pt")
with open("C:/Study Material/model/product_info.json", "r") as f:
    product_data = json.load(f)
    
# Global variables
detected_items = []
capture_history = []  # Track all captures
current_capture = None

def capture_single_image():
    """Capture a single image from camera"""
    try:
        # Try different camera backends for better compatibility
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            try:
                if backend:
                    cap = cv2.VideoCapture(0, backend)
                else:
                    cap = cv2.VideoCapture(0)
                
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        cap.release()
                        break
                cap.release()
            except:
                if cap:
                    cap.release()
                continue
        
        # Final capture
        if backend:
            cap = cv2.VideoCapture(0, backend)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            return None, "Could not open camera"
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Warm up camera - take a few frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.1)
        
        # Capture the actual image
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return frame, None
        else:
            return None, "Failed to capture image"
            
    except Exception as e:
        if cap:
            cap.release()
        return None, f"Camera error: {str(e)}"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture_image')
def capture_image():
    global detected_items, current_capture, capture_history
    
    # Capture image from camera
    frame, error = capture_single_image()
    
    if error:
        return jsonify({'error': error})
    
    if frame is None:
        return jsonify({'error': 'No frame captured'})
    
    # Save captured image
    timestamp = str(int(time.time()))
    save_path = f'static/captures/capture_{timestamp}.jpg'
    os.makedirs('static/captures', exist_ok=True)
    cv2.imwrite(save_path, frame)
    
    # Store current capture
    current_capture = {
        'path': save_path,
        'url': f'/{save_path}',
        'detected_items': []
    }
    capture_history.append(current_capture)
    
    # Run YOLO detection
    try:
        results = model(frame)[0]
        boxes = results.boxes
        temp_cart = defaultdict(int)
        
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                name = model.names[class_id]
                confidence = float(boxes.conf[i])
                
                # Only add items with confidence > 0.5
                if confidence > 0.5:
                    temp_cart[name] += 1
        
        # Add detected items to cart
        current_items = []
        for product, qty in temp_cart.items():
            price = product_data.get(product, {}).get("price", 0)
            img = product_data.get(product, {}).get("image", "")
            total = qty * price
            
            # Create item data
            item_data = {
                "name": product,
                "qty": qty,
                "price": price,
                "total": total,
                "image": img
            }
            
            # Add to current capture
            current_items.append(item_data)
            
            # Add to global cart
            existing_item = next((item for item in detected_items if item["name"] == product), None)
            
            if existing_item:
                existing_item["qty"] += qty
                existing_item["total"] = existing_item["qty"] * existing_item["price"]
            else:
                detected_items.append(item_data)
        
        # Update current capture with detected items
        current_capture['detected_items'] = current_items
        
        return jsonify({
            'status': 'success', 
            'items_detected': len(temp_cart),
            'capture_path': save_path,
            'capture_url': f'/{save_path}',
            'current_items': current_items
        })
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'})

@app.route('/get_capture_history')
def get_capture_history():
    history = [{
        'url': capture['url'],
        'items': capture['detected_items']
    } for capture in capture_history[-3:]]  # Show last 3 captures
    
    return jsonify(history)

@app.route('/get_latest_capture')
def get_latest_capture():
    if capture_history:
        return jsonify({'capture_url': capture_history[-1]['url']})
    return jsonify({'capture_url': None})

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/get_cart_items')
def get_cart_items():
    total = sum(item["total"] for item in detected_items)
    return jsonify({"items": detected_items, "total": total})

@app.route('/update_cart', methods=['POST'])
def update_cart():
    data = request.get_json()
    item_name = data.get('name')
    new_qty = int(data.get('qty', 0))
    
    for item in detected_items:
        if item['name'] == item_name:
            if new_qty <= 0:
                detected_items.remove(item)
            else:
                item['qty'] = new_qty
                item['total'] = item['qty'] * item['price']
            break
    
    return jsonify({'status': 'success'})

@app.route('/clear_cart')
def clear_cart():
    global detected_items, capture_history
    detected_items = []
    capture_history = []
    return jsonify({'status': 'success'})

@app.route('/payment')
def payment():
    total = sum(item["total"] for item in detected_items)
    return render_template('payment.html', total=total)

@app.route('/process_payment', methods=['POST'])
def process_payment():
    global detected_items, capture_history
    
    payment_method = request.form.get('payment_method')
    total_amount = sum(item["total"] for item in detected_items)
    
    # Clear cart after successful payment
    detected_items = []
    capture_history = []
    
    return render_template('payment_success.html', 
                         amount=total_amount, 
                         method=payment_method)

if __name__ == '__main__':
    app.run(debug=True)
