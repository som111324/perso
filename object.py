import torch
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use the appropriate YOLOv8 model weight file

# Initialize the DeepSORT object tracker
tracker = DeepSort(max_age=30, nn_budget=100, nms_max_overlap=1.0)

# Define class ID for "person"
PERSON_CLASS_ID = 0  # "person" class ID is usually 0 in YOLO models

# Video source (use 0 for webcam or replace with video file path)
video_source = "ABA Therapy_ Daniel - Communication.mp4"

# Capture video
cap = cv2.VideoCapture(video_source)

# Define codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLO
    results = model(frame, stream=True)

    # Initialize list to hold detection results
    detections = []

    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Get confidence score
            confidence = box.conf[0].item()
            # Get class id
            class_id = int(box.cls[0].item())

            # Filter detections: only keep "person"
            if class_id == PERSON_CLASS_ID:
                detections.append([[x1, y1, x2, y2], confidence, class_id])

    # Update tracker with filtered detections
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_idf
        ltrb = track.to_ltrb()

        # Draw bounding box
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        # Put track ID text
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Save the frame to output video
    out.write(frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
