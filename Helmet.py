import cv2
import numpy as np
import tensorflow as tf

# Load SSD MobileNetV2 COCO Model
MODEL_PATH = "D:/PythonProject1/ssd_mobilenet_v2_coco/saved_model"
model = tf.saved_model.load(MODEL_PATH)

# Get the model's inference function
infer = model.signatures["serving_default"]

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Expand dimensions

    # Perform inference
    detections = infer(input_tensor)

    # Extract detection results
    boxes = detections["detection_boxes"].numpy()[0]
    scores = detections["detection_scores"].numpy()[0]
    classes = detections["detection_classes"].numpy().astype(int)[0]

    height, width, _ = frame.shape

    # Loop through detections
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            y1, x1, y2, x2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {class_id}: {scores[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Helmet Detection - SSD MobileNetV2", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
