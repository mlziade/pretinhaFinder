from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train the model
results = model.train(
    data='training_pretinha/training_set_v1/data.yaml',  # path to your data.yaml file
    epochs=100,                       # number of training epochs
    imgsz=640,                        # image size (matches dataset preprocessing)
    batch=16,                         # batch size (reduce if you get memory errors)
    device='cpu',                     # '0' for GPU, 'cpu' for CPU
    patience=50,                      # early stopping patience
    save=True,                        # save checkpoints
    project='runs/train',             # where to save results
    name='pretinha_detector_v1',         # experiment name
)

print("Training complete!")
print(f"Best model saved at: runs/train/pretinha_detector/weights/best.pt")