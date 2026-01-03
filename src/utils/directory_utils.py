"""
Utility functions for directory and timestamp operations.
"""
import os
from datetime import datetime
from pathlib import Path


def ensure_directory(path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str or Path): Path to the directory
        
    Returns:
        Path: The Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamp(format_str="%Y%m%d_%H%M%S"):
    """
    Get current timestamp as a formatted string.
    
    Args:
        format_str (str): Format string for datetime formatting
        
    Returns:
        str: Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def get_data_directory():
    """
    Get the path to the data directory.
    
    Returns:
        Path: Path to the data directory
    """
    # Get the project root (parent of src/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "data"


def get_faces_directory():
    """
    Get the path to the faces directory.
    
    Returns:
        Path: Path to the faces directory
    """
    return get_data_directory() / "faces"


def list_subdirectories(path):
    """
    List all subdirectories in a given path.
    
    Args:
        path (str or Path): Path to search for subdirectories
        
    Returns:
        list: List of subdirectory Path objects
    """
    path = Path(path)
    if not path.exists():
        return []
    return [d for d in path.iterdir() if d.is_dir()]


def list_image_files(path, extensions=None):
    """
    List all image files in a given path.
    
    Args:
        path (str or Path): Path to search for images
        extensions (list): List of file extensions to include (default: common image formats)
        
    Returns:
        list: List of image file Path objects
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    path = Path(path)
    if not path.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(path.glob(f"*{ext}"))
        image_files.extend(path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)
