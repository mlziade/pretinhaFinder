import cv2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables
load_dotenv()

# Configuration from .env
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 5))
SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "dog_screenshots"))
SAVE_COOLDOWN = int(os.getenv("SAVE_COOLDOWN", 3))

# Create screenshots directory
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Use MobileNet SSD for object detection (lightweight and compatible)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PROTOTXT = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
MODEL = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

def download_model():
    """Download MobileNet SSD model files."""
    if not PROTOTXT.exists():
        print("Downloading model config...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
            str(PROTOTXT)
        )

    if not MODEL.exists():
        print("Downloading model weights (23MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/PINTO0309/MobileNet-SSD-RealSense/raw/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
            str(MODEL)
        )
        print("Model downloaded successfully!")

download_model()

# Load the model
net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))

# MobileNet SSD class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

DOG_CLASS_ID = CLASSES.index("dog")

def main():
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened. Press 'q' to quit.")
    print(f"Screenshots will be saved to: {SCREENSHOT_DIR.absolute()}")

    frame_count = 0
    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_count += 1

        # Detect objects every N frames
        if frame_count % FRAME_SKIP == 0:
            height, width = frame.shape[:2]

            # Prepare frame for MobileNet SSD
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Process detections
            dog_detected = False
            boxes = []
            confidences = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])

                # Check if it's a dog with sufficient confidence
                if class_id == DOG_CLASS_ID and confidence > 0.3:
                    dog_detected = True

                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    boxes.append([startX, startY, endX, endY])
                    confidences.append(float(confidence))

            # Draw boxes and save if dog detected
            display_frame = frame.copy()
            if dog_detected:
                # Draw bounding boxes
                for i, box in enumerate(boxes):
                    startX, startY, endX, endY = box
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Dog: {confidences[i]*100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 20
                    cv2.putText(display_frame, label, (startX, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save screenshot with cooldown
                current_time = datetime.now().timestamp()
                if current_time - last_save_time > SAVE_COOLDOWN:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    max_conf = max(confidences)
                    filename = SCREENSHOT_DIR / f"dog_{timestamp}_conf{max_conf:.2f}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"🐕 Dog detected! Screenshot saved: {filename}")
                    last_save_time = current_time

            cv2.imshow("Dog Detector", display_frame)
        else:
            cv2.imshow("Dog Detector", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    main()
