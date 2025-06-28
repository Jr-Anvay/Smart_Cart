# import cv2
# from ultralytics import YOLO
# import json
# from collections import defaultdict

# # Load the model
# model = YOLO("C:/Users/yadav/Desktop/smart_cart_project/updated_model/my_model/train/weights/best.pt")


# # Load product info
# with open("C:/Users/yadav/Desktop/smart_cart_project/updated_model/product_info.json", "r") as f:
#     product_data = json.load(f)

# # Setup quantity tracking
# product_count = defaultdict(int)

# # Open camera
# cap = cv2.VideoCapture(0)  # 0 = default webcam

# import time

# detected_in_session = set()
# frame_cooldown = 30  # Number of frames to wait before counting again

# frame_counter = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_counter += 1

#     results = model(frame)
#     detected_this_frame = set()

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             name = model.names[cls_id]
#             detected_this_frame.add(name)

#     # Count only new detections if enough time passed
#     if frame_counter % frame_cooldown == 0:
#         for item in detected_this_frame:
#             if item not in detected_in_session:
#                 product_count[item] += 1
#                 detected_in_session.add(item)

#         # Reset the session to allow re-counting after some time
#         detected_in_session.clear()

#     # Display frame
#     cv2.imshow("Live Feed - Press 'q' to stop", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
    
# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# # Generate bill
# print("\n----- Final List -----")
# total = 0
# for product, qty in product_count.items():
#     price = product_data.get(product, {}).get("price", 0)
#     cost = price * qty
#     print(f"{product}: Qty={qty}, Unit Price={price}, Total={cost}")
#     total += cost

# print(f"Total Bill: ₹{total}")


# import cv2
# import json
# from ultralytics import YOLO
# from collections import defaultdict
# from supervision import Detections  # or from supervision.detection.core import Detections
# from supervision.tracker.byte_tracker.core import ByteTrack

# # Load model and price data
# model = YOLO("C:/Users/yadav/Desktop/smart_cart_project/updated_model/my_model/train/weights/best.pt")


# with open("C:/Users/yadav/Desktop/smart_cart_project/updated_model/product_info.json", "r") as f:
#     product_data = json.load(f)

# # Set up
# cap = cv2.VideoCapture(0)
# tracker = ByteTrack()
# product_count = defaultdict(int)
# object_ids_seen = set()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]

#     # Convert YOLO results to Detections
#     detections = Detections(
#         xyxy=results.boxes.xyxy.cpu().numpy(),
#         confidence=results.boxes.conf.cpu().numpy(),
#         class_id=results.boxes.cls.cpu().numpy().astype(int)
#     )

#     # Track
#     tracks = tracker.update_with_detections(detections)
#     for track in tracks:
#         track_id, class_id, xyxy, conf = track
#         name = model.names[int(class_id)]

#     if track_id not in object_ids_seen:
#         object_ids_seen.add(track_id)
#         product_count[name] += 1

#     # Optional: draw box and label
#     x1, y1, x2, y2 = map(int, xyxy)
#     price = product_data.get(name, {}).get("price", 0)
#     label = f"{name} - ₹{price}"
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
# # Show video
#     cv2.imshow("Smart Cart Live - Press 'q' to stop", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# # Print Bill
# print("\n----- Final List -----")
# total = 0
# for product, qty in product_count.items():
#     price = product_data.get(product, {}).get("price", 0)
#     cost = price * qty
#     print(f"{product}: Qty={qty}, Unit Price={price}, Total={cost}")
#     total += cost
# print(f"Total Bill: ₹{total}")

import cv2
import json
from ultralytics import YOLO
from collections import defaultdict

# Load model and price data
model = YOLO("C:/Users/yadav/Desktop/smart_cart_project/updated_model/my_model/train/weights/best.pt")

with open("C:/Users/yadav/Desktop/smart_cart_project/updated_model/product_info.json", "r") as f:
    product_data = json.load(f)

# Set up
cap = cv2.VideoCapture(0)
product_count = defaultdict(int)
object_ids_seen = set()

# Tracking config
model.track(
    source=0,                             # Webcam
    show=True,                            # Show webcam window
    persist=True,                         # Keep track of objects across frames
    stream=True,                          # Yield frame-by-frame output
    tracker="bytetrack.yaml"              # Built-in ByteTrack config
)

# If you want to use this in a manual loop:
for result in model.track(source=0, persist=True, stream=True, tracker="bytetrack.yaml"):
    frame = result.orig_img
    boxes = result.boxes

    if boxes.id is None:
        continue

    for i in range(len(boxes)):
        track_id = int(boxes.id[i])
        class_id = int(boxes.cls[i])
        name = model.names[class_id]
        conf = float(boxes.conf[i])

        if track_id not in object_ids_seen:
            object_ids_seen.add(track_id)
            product_count[name] += 1

        # Draw box and label
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        price = product_data.get(name, {}).get("price", 0)
        label = f"{name} - ₹{price}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display
    cv2.imshow("Smart Cart Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release
cap.release()
cv2.destroyAllWindows()

# Print Bill
print("\n----- Final List -----")
total = 0
for product, qty in product_count.items():
    price = product_data.get(product, {}).get("price", 0)
    cost = price * qty
    print(f"{product}: Qty={qty}, Unit Price={price}, Total={cost}")
    total += cost
print(f"Total Bill: ₹{total}")

"C:/Study Material/model/my_model/train/weights/best.pt
"C:/Study Material/model/product_info.json

model = YOLO("C:/Study Material/model/my_model/train/weights/best.pt")
with open("C:/Study Material/model/product_info.json", "r") as f:
    product_data = json.load(f)

    <img src="/images/${item.image || 'placeholder.jpg'}" 
                                     alt="${item.name}" class="product-image">