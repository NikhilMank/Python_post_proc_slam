import cv2
import numpy as np
import yaml
from os import path as osp

def validate_file_path(file_path, valid_extensions):
    """Validates the file path and checks for valid extensions.

    Args:
        file_path (str): Path to the file
        valid_extensions (list): List of valid file extensions

    Returns:
        None

    Raises:
        ValueError: If the file does not exist or has an invalid extension
    """
    if not osp.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    if not any(file_path.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid file format for {file_path}. Expected one of {valid_extensions}.")


def load_image(image_path):
    """Loads an image from the given path as a grayscale image.

    Args:
        image_path (str): Path to the image file

    Returns:
        numpy.ndarray: Grayscale image
    """
    validate_file_path(image_path, ['.pgm'])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def load_metadata(yaml_path):
    """Loads metadata from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file

    Returns:
        dict: Metadata dictionary
    """
    validate_file_path(yaml_path, ['.yaml'])
    try:
        with open(yaml_path, 'r') as file:
            metadata = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {yaml_path}. Details: {e}")
    return metadata


def load_image_and_metadata(image_paths, yaml_path):
    """Loads images and metadata from the given paths.

    Args:
        image_paths (list): List of image file paths
        yaml_path (str): Path to the YAML metadata file

    Returns:
        tuple: (list of images, metadata dictionary)
    """
    if not isinstance(image_paths, list) or not all(isinstance(path, str) for path in image_paths):
        raise ValueError("image_paths must be a list of strings.")
    if not isinstance(yaml_path, str):
        raise ValueError("yaml_path must be a string.")

    images = [load_image(path) for path in image_paths]
    metadata = load_metadata(yaml_path)
    return images, metadata

def preprocess_image(image, occupied_thresh, free_thresh, negate):
    """Converts the image into a binary image based on the thresholds
    provided in the metadata.
    Makes the pixels above the occupied threshold white (255),
    below the free threshold black (0), and in between gray (127).

    Args:
        image (numpy.ndarray): grayscale image
        occupied_thresh (float): The threshold for occupied pixels
        free_thresh (float): The threshold for free pixels
        negate (bool): Whether to negate the image

    Returns:
       numpy.ndarray: binary image based on the thresholds
    """
    binary_image = np.zeros_like(image) # Create a blank image
    # Convert image based on occupied and free thresholds
    binary_image[image > occupied_thresh * 255] = 255  # Occupied
    binary_image[image < free_thresh * 255] = 0        # Free
    binary_image[(image >= free_thresh * 255) & (image <= occupied_thresh * 255)] = 127  # Unknown

    if negate:  
        binary_image = 255 - binary_image   # Invert the image

    return binary_image