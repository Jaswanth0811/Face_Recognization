"""
Face database loader that builds face encodings from data/faces/ directory.

Expected directory structure:
data/faces/
    person1/
        image1.jpg
        image2.jpg
    person2/
        image1.jpg
        image2.jpg
"""
import face_recognition
import numpy as np
from pathlib import Path
from utils.directory_utils import get_faces_directory, list_subdirectories, list_image_files


class FaceDatabase:
    """
    Face database that loads and manages face encodings.
    """
    
    def __init__(self, faces_dir=None):
        """
        Initialize the face database.
        
        Args:
            faces_dir (str or Path): Path to the faces directory. 
                                     If None, uses default data/faces/ directory.
        """
        self.faces_dir = Path(faces_dir) if faces_dir else get_faces_directory()
        self.known_face_encodings = []
        self.known_face_names = []
        self.encoding_count = {}
        
    def load_database(self):
        """
        Load all face encodings from the faces directory.
        
        Returns:
            tuple: (known_face_encodings, known_face_names)
        """
        print(f"Loading face database from: {self.faces_dir}")
        
        if not self.faces_dir.exists():
            print(f"Warning: Faces directory does not exist: {self.faces_dir}")
            return self.known_face_encodings, self.known_face_names
        
        # Get all person directories
        person_dirs = list_subdirectories(self.faces_dir)
        
        if not person_dirs:
            print(f"Warning: No person directories found in {self.faces_dir}")
            return self.known_face_encodings, self.known_face_names
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            image_files = list_image_files(person_dir)
            
            if not image_files:
                print(f"Warning: No images found for {person_name}")
                continue
            
            print(f"Loading images for {person_name}: {len(image_files)} images")
            encodings_loaded = 0
            
            for i, image_path in enumerate(image_files, 1):
                print(f"  Processing {person_name} ({i}/{len(image_files)}): {image_path.name}", end='\r')
                encoding = self._load_face_encoding(image_path, person_name)
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(person_name)
                    encodings_loaded += 1
            
            print(f"  Loaded {encodings_loaded} encodings for {person_name}                    ")
            self.encoding_count[person_name] = encodings_loaded
        
        print(f"\nTotal faces loaded: {len(self.known_face_encodings)}")
        print(f"Unique persons: {len(self.encoding_count)}")
        
        return self.known_face_encodings, self.known_face_names
    
    def _load_face_encoding(self, image_path, person_name):
        """
        Load a single face encoding from an image file.
        
        Args:
            image_path (Path): Path to the image file
            person_name (str): Name of the person in the image
            
        Returns:
            numpy.ndarray or None: Face encoding array, or None if no face found
        """
        try:
            # Load the image
            image = face_recognition.load_image_file(str(image_path))
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"  Warning: No face found in {image_path.name}")
                return None
            
            if len(encodings) > 1:
                print(f"  Warning: Multiple faces found in {image_path.name}, using first one")
            
            return encodings[0]
            
        except Exception as e:
            print(f"  Error loading {image_path.name}: {str(e)}")
            return None
    
    def get_encodings(self):
        """
        Get the loaded face encodings and names.
        
        Returns:
            tuple: (known_face_encodings, known_face_names)
        """
        return self.known_face_encodings, self.known_face_names
    
    def get_summary(self):
        """
        Get a summary of the loaded database.
        
        Returns:
            dict: Summary information about loaded faces
        """
        return {
            'total_encodings': len(self.known_face_encodings),
            'unique_persons': len(self.encoding_count),
            'encoding_count': self.encoding_count,
            'faces_directory': str(self.faces_dir)
        }
