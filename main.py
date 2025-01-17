#!/usr/bin/env python3
from tkinter import Tk
from tkinter import filedialog

from cv2 import imread, calcHist, normalize, compareHist, IMREAD_GRAYSCALE, HISTCMP_BHATTACHARYYA
from os import listdir, path, chdir, makedirs

from shutil import move

from numpy import array
from tqdm import tqdm
from collections import defaultdict

from PIL.Image import frombytes

def select_image_folder():

    folder_path = filedialog.askdirectory(title="Select Image Folder")

    return folder_path


def get_image_files():
    # Define supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    # Get all files in the folder and filter only images
    image_files = [
        path.join(file)
        for file in listdir()
        if file.lower().endswith(image_extensions)
    ]
    return image_files


def compute_histogram(image_path):
    """
    Compute the normalized grayscale histogram of an image.
    """
    image = imread(image_path, IMREAD_GRAYSCALE)
    if image is None:
        return None  # Return None if the image can't be read
    hist = calcHist([image], [0], None, [256], [0, 256])
    return normalize(hist, hist).flatten()


def compute_histogram(image, num_buckets=256):
    """
    Compute the normalized grayscale histogram of an image.
    Args:
        image (np.ndarray): Input grayscale image.
        num_buckets (int): Number of histogram bins.
    Returns:
        np.ndarray: Normalized histogram.
    """
    hist = calcHist([image], [0], None, [num_buckets], [0, 256])
    return normalize(hist, hist).flatten()


def group_images_by_size(image_paths):
    """
    Group images by their size to reduce unnecessary comparisons.
    Args:
        image_paths (list): List of image paths.
    Returns:
        dict: Groups of image paths keyed by image dimensions.
    """
    size_groups = defaultdict(list)
    
    # Add tqdm progress bar
    for img_path in tqdm(image_paths, desc="Grouping Images by Size", unit="image"):
        img = imread(img_path, IMREAD_GRAYSCALE)
        if img is None:
            continue
        size_groups[img.shape].append(img_path)
    
    return size_groups

def read_image(file_path):
    # Handle standard formats using OpenCV
    return imread(file_path, IMREAD_GRAYSCALE)

def compare_images_histograms(image_paths, threshold=0.99):
    """
    Compare images using histogram similarity without grouping by size.
    Args:
        image_paths (list): List of image file paths.
        threshold (float): Similarity threshold for considering duplicates.
    Returns:
        dict: Dictionary where each key is an image, and its value is a list of similar images.
    """
    similar_dict = {}

    # Track histogram computation and comparison
    histograms = []
    for img_path in tqdm(image_paths, desc="Computing Histograms", unit="image"):
        img = imread(img_path, IMREAD_GRAYSCALE)
        if img is None:
            continue
        hist = compute_histogram(img)
        histograms.append((img_path, hist))

    # Compare histograms
    for i in tqdm(range(len(histograms)), desc="Comparing Images", unit="image"):
        current_image, current_hist = histograms[i]
        current_group = []
        for j in range(i + 1, len(histograms)):
            other_image, other_hist = histograms[j]

            # Use Bhattacharyya distance for comparison
            similarity = 1 - compareHist(current_hist, other_hist, HISTCMP_BHATTACHARYYA)
            if similarity > threshold:
                current_group.append(other_image)

        if current_group:
            similar_dict[current_image] = current_group

    return similar_dict



def move_duplicate_images(similar_dict, destination_folder="!duplicate_images"):
    """
    Move duplicate images (excluding keys) to a separate folder.

    Args:
        similar_dict (dict): Dictionary where keys are reference images,
                             and values are lists of duplicate image paths.
        destination_folder (str): Name of the folder to move duplicates into.
    """
    if(not similar_dict): 
        return
    if not path.exists(destination_folder):
            makedirs(destination_folder)

    print(f"Moving duplicate images to: {destination_folder}")

    # Track moved files to prevent duplicates
    moved_files = set()

    for key, duplicates in similar_dict.items():
        for duplicate in duplicates:
            if duplicate in moved_files:
                # Skip if the file has already been moved
                continue

            # Get the base name of the file (e.g., 'image.jpg')
            file_name = path.basename(duplicate)

            # Generate a unique destination path
            destination_path = path.join(destination_folder, file_name)
            counter = 1
            while destination_path in moved_files or path.exists(destination_path):
                # If file already exists, append a counter to the file name
                file_name_no_ext, file_ext = path.splitext(file_name)
                destination_path = path.join(destination_folder, f"{file_name_no_ext}_{counter}{file_ext}")
                counter += 1

            try:
                # Move the file
                move(duplicate, destination_path)
                moved_files.add(duplicate)  # Track the moved file
                print(f"Moved: {duplicate} -> {destination_path}")
            except FileNotFoundError:
                print(f"File not found: {duplicate}, skipping.")

    print("Duplicate images have been moved.")


def main():
    # root = Tk()
    image_folder_path = select_image_folder()
    print("image_folder_path ", image_folder_path)
    chdir(image_folder_path)

    image_files = get_image_files()
    print("Images found ", len(image_files))

    similar_groups = compare_images_histograms(image_files)
    print("Similar images ", similar_groups)

    move_duplicate_images(similar_groups)

if __name__ == "__main__":
    main()
