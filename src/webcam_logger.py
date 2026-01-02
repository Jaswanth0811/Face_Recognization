"""
Face Recognition Webcam Logger

A starter scaffold for real-time face recognition using webcam.
This module captures video from the webcam, detects faces, and logs recognized faces.
"""
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from utils.directory_utils import ensure_directory, get_timestamp
from face_database import FaceDatabase


class FaceRecognitionLogger:
    """
    Webcam-based face recognition logger.
    """
    
    def __init__(self, faces_dir=None, log_dir=None):
        """
        Initialize the face recognition logger.
        
        Args:
            faces_dir (str or Path): Path to the faces database directory
            log_dir (str or Path): Path to the directory for logs
        """
        self.face_database = FaceDatabase(faces_dir)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        ensure_directory(self.log_dir)
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Video capture settings
        self.video_capture = None
        self.process_every_n_frames = 2  # Process every nth frame for performance
        self.frame_count = 0
        
        # Recognition settings
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Logging
        self.log_file = None
        self.recognized_faces = set()
        
    def load_known_faces(self):
        """
        Load known faces from the database.
        
        Returns:
            bool: True if faces were loaded successfully
        """
        print("Loading known faces database...")
        self.known_face_encodings, self.known_face_names = self.face_database.load_database()
        
        if len(self.known_face_encodings) == 0:
            print("Warning: No known faces loaded. Please add faces to data/faces/ directory.")
            print("Expected structure: data/faces/person_name/image.jpg")
            return False
        
        summary = self.face_database.get_summary()
        print(f"\nDatabase loaded successfully!")
        print(f"Total encodings: {summary['total_encodings']}")
        print(f"Unique persons: {summary['unique_persons']}")
        
        return True
    
    def initialize_camera(self, camera_index=0):
        """
        Initialize the video camera.
        
        Args:
            camera_index (int): Index of the camera to use (default: 0)
            
        Returns:
            bool: True if camera was initialized successfully
        """
        print(f"Initializing camera {camera_index}...")
        self.video_capture = cv2.VideoCapture(camera_index)
        
        if not self.video_capture.isOpened():
            print("Error: Could not open camera")
            return False
        
        print("Camera initialized successfully")
        return True
    
    def initialize_logging(self):
        """
        Initialize the logging system.
        """
        timestamp = get_timestamp()
        log_filename = self.log_dir / f"face_log_{timestamp}.txt"
        self.log_file = open(log_filename, 'w')
        self.log_file.write(f"Face Recognition Log - {timestamp}\n")
        self.log_file.write("=" * 50 + "\n\n")
        print(f"Logging to: {log_filename}")
    
    def log_recognition(self, name):
        """
        Log a face recognition event.
        
        Args:
            name (str): Name of the recognized person
        """
        if name != "Unknown" and name not in self.recognized_faces:
            timestamp = get_timestamp("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Recognized: {name}\n"
            self.log_file.write(log_entry)
            self.log_file.flush()
            self.recognized_faces.add(name)
            print(log_entry.strip())
    
    def process_frame(self, frame):
        """
        Process a single video frame for face recognition.
        
        Args:
            frame (numpy.ndarray): Video frame from the camera
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Only process every nth frame
        if self.frame_count % self.process_every_n_frames == 0:
            # Find all faces and face encodings in the current frame
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        self.log_recognition(name)
                
                self.face_names.append(name)
        
        self.frame_count += 1
        
        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since we detected on a scaled down frame
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Run the face recognition logger.
        
        Args:
            camera_index (int): Index of the camera to use
        """
        # Load known faces
        if not self.load_known_faces():
            print("\nPlease add face images to the data/faces/ directory and try again.")
            print("Directory structure: data/faces/person_name/image.jpg")
            return
        
        # Initialize camera
        if not self.initialize_camera(camera_index):
            return
        
        # Initialize logging
        self.initialize_logging()
        
        print("\nStarting face recognition...")
        print("Press 'q' to quit")
        print("-" * 50)
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.video_capture.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the resulting frame
                cv2.imshow('Face Recognition Logger', processed_frame)
                
                # Hit 'q' on the keyboard to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources.
        """
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        
        if self.log_file:
            self.log_file.write("\n" + "=" * 50 + "\n")
            self.log_file.write(f"Session ended - {get_timestamp('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
        
        print("Resources cleaned up")


def main():
    """
    Main entry point for the face recognition logger.
    """
    print("=" * 50)
    print("Face Recognition Webcam Logger")
    print("=" * 50)
    print()
    
    logger = FaceRecognitionLogger()
    logger.run()


if __name__ == "__main__":
    main()
