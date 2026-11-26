
from ultralytics import YOLO
import sqlite3
import uuid
from datetime import datetime
import pytz
import cv2
import os


def insert_object_occurrence(object_class, frame_id, conn, cursor):
    # Get current local time
    time_now = datetime.now()
    timestamp = time_now.isoformat()

    # Insert object occurrence into database
    cursor.execute('''
        INSERT INTO object_occurrences (frame_id, object_class, timestamp)
        VALUES (?, ?, ?)
    ''', (frame_id, object_class, timestamp))

    conn.commit()


def save_frame_as_jpg(frame, name):
    os.makedirs('snapshots', exist_ok=True)

    filename = os.path.join('snapshots', f'{name}.jpg')
    cv2.imwrite(filename, frame)

    return filename


# Real-time object detection and database insertion
def inference(model_version='v2', video_source=None, frame_count=100):

    # Initialize YOLO model
    model = YOLO(f'{model_version}/weights/best.pt')

    print("YOLO Model Loaded:", model_version)

    # Connect to SQLite database
    conn = sqlite3.connect('object_occurrences.db')
    cursor = conn.cursor()

    # Ensure database table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_occurrences (
            frame_id TEXT,
            object_class TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()

    detected_objects = []
    annotated_frame = None

    def process_frame(frame):
        # Perform object detection
        results = model.predict(
            frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        frame_detections = []
        annotated = results[0].plot()

        for result in results:
            for box in result.boxes:

                label = int(box.cls[0].item())
                class_name = model.names[label]
                confidence = float(box.conf[0].item())

                frame_id = str(uuid.uuid4())

                # Save annotated detection frame
                save_frame_as_jpg(annotated, frame_id)

                # Save occurrence to database
                insert_object_occurrence(
                    class_name,
                    frame_id,
                    conn,
                    cursor
                )

                detection = {
                    'object': class_name,
                    'confidence': confidence,
                    'frame_id': frame_id
                }

                frame_detections.append(detection)

                print(
                    f"Object Found: {class_name} "
                    f"| Confidence: {confidence:.2%}"
                )

        return annotated, frame_detections

    try:
        # Check if uploaded source is an image
        is_image = (
            isinstance(video_source, str)
            and video_source.lower().endswith(
                ('.jpg', '.jpeg', '.png')
            )
        )

        if is_image:
            # Read uploaded image
            frame = cv2.imread(video_source)

            if frame is None:
                raise ValueError(
                    f"Unable to read image: {video_source}"
                )

            annotated_frame, frame_detections = process_frame(frame)
            detected_objects.extend(frame_detections)

        else:
            # Webcam or uploaded video
            source = video_source if video_source is not None else 0

            if isinstance(source, str) and source.isdigit():
                source = int(source)

            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                raise RuntimeError(
                    f"Unable to open video source: {source}"
                )

            try:
                frame_c = frame_count

                while cap.isOpened() and frame_c > 0:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    annotated_frame, frame_detections = process_frame(frame)

                    detected_objects.extend(frame_detections)

                    frame_c -= 1

            finally:
                cap.release()
                cv2.destroyAllWindows()

        if not detected_objects:
            print("No objects detected.")

        return {
            'detections': detected_objects,
            'frame': annotated_frame
        }

    finally:
        cursor.close()
        conn.close()
