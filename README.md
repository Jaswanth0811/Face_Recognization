# Face Recognition Webcam Logger

A Python-based face recognition system that uses webcam for real-time face detection and logging.

## Features

- Real-time face recognition using webcam
- Face database loader that builds encodings from images
- Automatic logging of recognized faces with timestamps
- Visual feedback with bounding boxes and names
- Utility functions for directory management and timestamps

## Project Structure

```
Face_Recognization/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── directory_utils.py    # Directory and timestamp utilities
│   ├── face_database.py           # Face encoding database loader
│   └── webcam_logger.py           # Main webcam logger application
├── data/
│   └── faces/                     # Face database directory
│       ├── person1/               # Create folders for each person
│       │   ├── image1.jpg
│       │   └── image2.jpg
│       └── person2/
│           └── image1.jpg
├── logs/                          # Generated logs directory
├── requirements.txt               # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaswanth0811/Face_Recognization.git
```
```bash
cd Face_Recognization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The `face-recognition` library requires `dlib` which may need additional setup on some systems. See [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation) for details.

## Usage

### 1. Prepare Face Database

Create directories for each person in `data/faces/` and add their photos:

```
data/faces/
├── john/
│   ├── john1.jpg
│   └── john2.jpg
└── jane/
    └── jane1.jpg
```

**Tips:**
- Use clear, well-lit photos
- One face per image works best
- Multiple photos per person improves recognition accuracy
- Support formats: .jpg, .jpeg, .png, .bmp, .gif

### 2. Run the Webcam Logger

```bash
cd src
```
```bash
python webcam_logger.py
```

The application will:
1. Load all faces from `data/faces/`
2. Start the webcam
3. Detect and recognize faces in real-time
4. Log recognized faces to `logs/face_log_[timestamp].txt`
5. Display video with bounding boxes and names

Press 'q' to quit.

## Module Documentation

### `utils/directory_utils.py`

Utilities for directory and timestamp operations:
- `ensure_directory(path)` - Create directory if it doesn't exist
- `get_timestamp(format_str)` - Get formatted timestamp string
- `get_data_directory()` - Get path to data directory
- `get_faces_directory()` - Get path to faces directory
- `list_subdirectories(path)` - List all subdirectories
- `list_image_files(path)` - List all image files

### `face_database.py`

Face database loader:
- `FaceDatabase` - Class for loading and managing face encodings
- `load_database()` - Load all face encodings from faces directory
- `get_encodings()` - Get loaded encodings and names
- `get_summary()` - Get database summary

### `webcam_logger.py`

Main webcam logger application:
- `FaceRecognitionLogger` - Main logger class
- `load_known_faces()` - Load face database
- `initialize_camera()` - Initialize webcam
- `process_frame()` - Process video frame for recognition
- `run()` - Run the logger application

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- face_recognition
- numpy
- Pillow

## Troubleshooting

### No camera detected
- Check if your camera is connected and working
- Try a different camera index in the code (0, 1, 2, etc.)

### No faces loaded
- Ensure images are in `data/faces/person_name/` structure
- Check that image files are valid
- Look at console output for specific loading errors

### Poor recognition accuracy
- Add more images per person (3-5 recommended)
- Use clear, well-lit photos
- Ensure faces are clearly visible
- Try adjusting camera angle and lighting

## License

This project is open source and available under the MIT License.
