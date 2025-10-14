# Pretinha Finder

This project is a simple application to recognize dogs in a live camera feed that i used to try out and learn more about YOLOv8 and object detection.

## Features
- Live camera feed processing
- Dog detection using YOLOv8 (but you can change the class_id to detect other objects, use the `yolo predict` command to find out the class_id of other objects)
- Bounding box drawing around detected objects
- Prints from the camera feed and saves the images when a dog is detected