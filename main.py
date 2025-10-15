import cv2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
from ultralytics import YOLO

load_dotenv()

# Configuration
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 5))
YOLO_MODEL = os.getenv("YOLO_MODEL", "runs/train/pretinha_detector_v12/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "dog_screenshots"))
SAVE_COOLDOWN = int(os.getenv("SAVE_COOLDOWN", 3))

# Create screenshots and models folders
SCREENSHOT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load custom trained model directly
print(f"Loading custom YOLOv8 model: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)

PRETINHA_CLASS_ID = 0  # Custom model has only one class: 'pretinha'

def main():
    # Open camera with DirectShow backend (more reliable on Windows)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera with DirectShow, trying default...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

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
            # Run YOLOv8 inference
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Process detections
            pretinha_detected = False
            confidences = []

            # Get the annotated frame with bounding boxes
            annotated_frame = results[0].plot()

            # Check if Pretinha was detected
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == PRETINHA_CLASS_ID:
                        pretinha_detected = True
                        confidences.append(confidence)

            # Save screenshot if Pretinha detected
            if pretinha_detected:
                current_time = datetime.now().timestamp()
                if current_time - last_save_time > SAVE_COOLDOWN:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    max_conf = max(confidences)
                    filename = SCREENSHOT_DIR / f"pretinha_{timestamp}_conf{max_conf:.2f}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"Pretinha detected!!! Screenshot saved: {filename}")
                    last_save_time = current_time

            cv2.imshow("Pretinha Detector - Custom YOLOv8", annotated_frame)
        else:
            cv2.imshow("Pretinha Detector - Custom YOLOv8", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    main()
