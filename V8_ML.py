# ================================================================
# IP CAMERA STORY NARRATOR - REAL-TIME JSON OUTPUT
# ================================================================
# Processes IP camera frames and outputs story segments in JSON format every 20 seconds

import cv2
import json
import time
import threading
import os
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
import requests
import numpy as np

class IPCameraStoryNarrator:
    def __init__(self, camera_ip, model_path='yolov8n.pt', json_folder='story_segments'):
        self.camera_ip = camera_ip
        self.model = YOLO(model_path)
        self.story_segments = []
        self.narration_interval = 20.0  # 20 seconds
        self.last_narration_time = 0
        self.detection_history = []
        self.current_detections = []
        self.frame_width = 0
        self.frame_height = 0
        self.running = False

        # JSON folder setup
        self.json_folder = json_folder
        self.setup_json_folder()

        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = os.path.join(self.json_folder, f"session_{self.session_id}")
        os.makedirs(self.session_folder, exist_ok=True)

        # Color palette for different object classes
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        self.class_colors = {}

        print(f"📁 JSON output folder: {self.json_folder}")
        print(f"📁 Session folder: {self.session_folder}")

    def setup_json_folder(self):
        """Create JSON folder structure"""
        try:
            # Create main JSON folder
            os.makedirs(self.json_folder, exist_ok=True)

            # Create subfolders
            subfolders = ['daily', 'sessions', 'latest']
            for subfolder in subfolders:
                os.makedirs(os.path.join(self.json_folder, subfolder), exist_ok=True)

            print(f"✅ JSON folder structure created: {self.json_folder}")

        except Exception as e:
            print(f"❌ Failed to create JSON folder: {e}")

    def save_segment_to_json(self, story_data):
        """Save story segment to multiple JSON files"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Save individual segment file
            segment_filename = f"segment_{story_data['segment_id']:03d}_{timestamp_str}.json"
            segment_path = os.path.join(self.session_folder, segment_filename)

            with open(segment_path, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)

            # 2. Save to daily summary file
            daily_filename = f"daily_{datetime.now().strftime('%Y%m%d')}.json"
            daily_path = os.path.join(self.json_folder, 'daily', daily_filename)

            # Load existing daily data or create new
            daily_data = {"date": datetime.now().strftime('%Y-%m-%d'), "segments": []}
            if os.path.exists(daily_path):
                with open(daily_path, 'r', encoding='utf-8') as f:
                    daily_data = json.load(f)

            daily_data["segments"].append(story_data)

            with open(daily_path, 'w', encoding='utf-8') as f:
                json.dump(daily_data, f, indent=2, ensure_ascii=False)

            # 3. Save as latest segment
            latest_path = os.path.join(self.json_folder, 'latest', 'latest_segment.json')
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)

            # 4. Update session summary
            session_summary_path = os.path.join(self.session_folder, 'session_summary.json')
            session_summary = {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "camera_ip": self.camera_ip,
                "total_segments": len(self.story_segments),
                "segments": self.story_segments
            }

            with open(session_summary_path, 'w', encoding='utf-8') as f:
                json.dump(session_summary, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved to: {segment_path}")
            print(f"💾 Updated: {daily_path}")
            print(f"💾 Latest: {latest_path}")

        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")

    def get_color_for_class(self, class_name):
        """Assign consistent colors to object classes"""
        if class_name not in self.class_colors:
            color_index = len(self.class_colors) % len(self.color_palette)
            self.class_colors[class_name] = self.color_palette[color_index]
        return self.class_colors[class_name]

    def get_position_description(self, bbox, frame_width, frame_height):
        """Get human-readable position description for detected objects"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Horizontal position
        if center_x < frame_width * 0.33:
            h_pos = "left side"
        elif center_x > frame_width * 0.66:
            h_pos = "right side"
        else:
            h_pos = "center"

        # Vertical position
        if center_y < frame_height * 0.33:
            v_pos = "top"
        elif center_y > frame_height * 0.66:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Combine positions
        if h_pos == "center" and v_pos == "middle":
            return "center of frame"
        elif h_pos == "center":
            return f"{v_pos} center"
        elif v_pos == "middle":
            return f"{h_pos} of frame"
        else:
            return f"{v_pos} {h_pos}"

    def should_create_story_segment(self, current_time):
        """Check if it's time to create a new story segment (every 20 seconds)"""
        return current_time - self.last_narration_time >= self.narration_interval

    def create_story_segment(self, timestamp, detections):
        """Create a story segment based on current detections"""
        if not detections:
            story_text = f"The camera shows a quiet scene with no objects detected."
            story_data = {
                "timestamp": timestamp,
                "segment_id": len(self.story_segments) + 1,
                "story_text": story_text,
                "detected_objects": [],
                "total_objects": 0,
                "scene_activity": "low"
            }
        else:
            # Group detections by class
            class_info = defaultdict(list)
            for det in detections:
                class_info[det['class']].append({
                    'position': det['position'],
                    'confidence': det['confidence']
                })

            # Create narrative descriptions
            descriptions = []
            detected_objects = []

            for class_name, instances in class_info.items():
                obj_data = {
                    "class": class_name,
                    "count": len(instances),
                    "positions": [inst['position'] for inst in instances],
                    "avg_confidence": sum(inst['confidence'] for inst in instances) / len(instances)
                }
                detected_objects.append(obj_data)

                if len(instances) == 1:
                    conf = instances[0]['confidence']
                    pos = instances[0]['position']
                    confidence_desc = "clearly visible" if conf > 0.8 else "detected" if conf > 0.5 else "faintly visible"
                    descriptions.append(f"a {class_name} {confidence_desc} on the {pos}")
                else:
                    avg_conf = sum(inst['confidence'] for inst in instances) / len(instances)
                    confidence_desc = "clearly visible" if avg_conf > 0.8 else "detected" if avg_conf > 0.5 else "faintly visible"
                    descriptions.append(f"{len(instances)} {class_name}s {confidence_desc}")

            # Create story text
            if len(descriptions) == 1:
                story_text = f"The camera captures {descriptions[0]}."
            elif len(descriptions) == 2:
                story_text = f"The scene shows {descriptions[0]} and {descriptions[1]}."
            else:
                story_text = f"The scene contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}."

            # Determine scene activity level
            total_objects = len(detections)
            if total_objects >= 5:
                activity = "high"
            elif total_objects >= 2:
                activity = "medium"
            else:
                activity = "low"

            story_data = {
                "timestamp": timestamp,
                "segment_id": len(self.story_segments) + 1,
                "story_text": story_text,
                "detected_objects": detected_objects,
                "total_objects": total_objects,
                "scene_activity": activity,
                "detection_details": detections,
                "created_at": datetime.now().isoformat(),
                "camera_ip": self.camera_ip,
                "session_id": self.session_id
            }

        self.story_segments.append(story_data)
        self.last_narration_time = timestamp

        # Save to JSON files
        self.save_segment_to_json(story_data)

        # Output JSON to console/log
        json_output = json.dumps(story_data, indent=2)
        print(f"\n{'='*60}")
        print(f"📖 STORY SEGMENT {story_data['segment_id']} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(json_output)
        print(f"{'='*60}\n")

        return story_data

    def process_frame(self, frame):
        """Process a single frame and extract detections"""
        if frame is None:
            return []

        results = self.model.predict(source=frame, conf=0.3, verbose=False)
        detections = []

        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                cls_name = results[0].names[cls]

                detection = {
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': coords.tolist(),
                    'position': self.get_position_description(coords, self.frame_width, self.frame_height)
                }
                detections.append(detection)

        return detections

    def connect_to_camera(self):
        """Connect to IP camera stream"""
        try:
            # Try different URL formats for IP cameras
            possible_urls = [
                f"http://{self.camera_ip}/video",
                f"http://{self.camera_ip}:4747/video",
                f"http://{self.camera_ip}/mjpeg",
                f"rtsp://{self.camera_ip}/stream",
                self.camera_ip  # Direct URL
            ]

            cap = None
            for url in possible_urls:
                print(f"🔗 Trying to connect to: {url}")
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    print(f"✅ Successfully connected to: {url}")
                    break
                cap.release()

            if not cap or not cap.isOpened():
                print(f"❌ Failed to connect to camera: {self.camera_ip}")
                return None

            # Get frame dimensions
            ret, frame = cap.read()
            if ret:
                self.frame_height, self.frame_width = frame.shape[:2]
                print(f"📐 Frame dimensions: {self.frame_width}x{self.frame_height}")

            return cap

        except Exception as e:
            print(f"❌ Camera connection error: {e}")
            return None

    def run_story_narrator(self):
        """Main loop for IP camera story narration"""
        print("🤖 IP CAMERA STORY NARRATOR STARTING...")
        print(f"📹 Camera IP: {self.camera_ip}")
        print(f"⏱️ Story segments every {self.narration_interval} seconds")
        print("=" * 60)

        cap = self.connect_to_camera()
        if not cap:
            return

        self.running = True
        start_time = time.time()
        frame_count = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break

                current_time = time.time() - start_time
                frame_count += 1

                # Process frame for object detection
                detections = self.process_frame(frame)
                self.current_detections = detections

                # Check if it's time to create a story segment
                if self.should_create_story_segment(current_time):
                    story_segment = self.create_story_segment(current_time, detections)

                    # Here you could also send the JSON to an API endpoint:
                    # self.send_to_api(story_segment)

                # Optional: Display frame with detections (for debugging)
                # self.draw_detections(frame, detections)
                # cv2.imshow('IP Camera Feed', frame)

                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

                # Press 'q' to quit (if displaying video)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        except KeyboardInterrupt:
            print("\n🛑 Stopping camera narrator...")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()

            # Create final session summary
            self.create_final_session_summary()
            print("✅ Camera narrator stopped")

    def create_final_session_summary(self):
        """Create final session summary with statistics"""
        try:
            session_stats = {
                "session_id": self.session_id,
                "start_time": self.session_id,  # Using session_id as start time indicator
                "end_time": datetime.now().isoformat(),
                "camera_ip": self.camera_ip,
                "total_segments": len(self.story_segments),
                "total_runtime_seconds": time.time() if hasattr(self, 'start_time') else 0,
                "segments_summary": {
                    "high_activity": len([s for s in self.story_segments if s.get('scene_activity') == 'high']),
                    "medium_activity": len([s for s in self.story_segments if s.get('scene_activity') == 'medium']),
                    "low_activity": len([s for s in self.story_segments if s.get('scene_activity') == 'low'])
                },
                "most_detected_objects": self.get_most_detected_objects(),
                "segments": self.story_segments
            }

            final_summary_path = os.path.join(self.session_folder, 'final_session_summary.json')
            with open(final_summary_path, 'w', encoding='utf-8') as f:
                json.dump(session_stats, f, indent=2, ensure_ascii=False)

            print(f"📊 Final session summary saved: {final_summary_path}")

        except Exception as e:
            print(f"❌ Failed to create final summary: {e}")

    def get_most_detected_objects(self):
        """Get statistics of most frequently detected objects"""
        object_counts = {}
        for segment in self.story_segments:
            for obj in segment.get('detected_objects', []):
                class_name = obj['class']
                object_counts[class_name] = object_counts.get(class_name, 0) + obj['count']

        # Sort by count and return top 5
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_objects[:5])

    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame (optional, for debugging)"""
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = self.get_color_for_class(det['class'])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def send_to_api(self, story_segment):
        """Send story segment JSON to an API endpoint (optional)"""
        try:
            # Replace with your API endpoint
            api_url = "http://your-api-endpoint.com/story-segments"
            headers = {'Content-Type': 'application/json'}

            response = requests.post(api_url, json=story_segment, headers=headers)
            if response.status_code == 200:
                print(f"✅ Story segment sent to API successfully")
            else:
                print(f"❌ API request failed: {response.status_code}")
        except Exception as e:
            print(f"❌ API send error: {e}")

    def stop(self):
        """Stop the narrator"""
        self.running = False

# ================================================================
# USAGE EXAMPLE
# ================================================================

def main():
    # Replace with your camera IP address
    camera_ip = "http://192.168.1.4:4747/video"  # Example IP
    # Or use full URL: "http://192.168.1.100:8080/video"

    # Custom JSON folder path (optional)
    json_folder = "story_segments"  # or use full path like "/path/to/your/json_folder"

    narrator = IPCameraStoryNarrator(camera_ip, json_folder=json_folder)

    try:
        narrator.run_story_narrator()
    except KeyboardInterrupt:
        print("\nStopping...")
        narrator.stop()

if __name__ == "__main__":
    main()
