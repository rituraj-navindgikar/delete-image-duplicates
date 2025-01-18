#!/usr/bin/env python3
from tkinter import filedialog
from cv2 import imread, calcHist, normalize, compareHist, resize, IMREAD_GRAYSCALE, HISTCMP_BHATTACHARYYA, INTER_AREA
from os import listdir, path, chdir, makedirs
from shutil import move
from tqdm import tqdm

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

def preprocess_image(image, size=(256, 256)):
    """
    Preprocess the image by resizing to a fixed size.
    Args:
        image (np.ndarray): Input grayscale image.
        size (tuple): Target size (width, height) for resizing.
    Returns:
        np.ndarray: Preprocessed image.
    """
    return resize(image, size, interpolation=INTER_AREA)

def compute_histogram(image, num_buckets=256):
    """
    Compute the normalized grayscale histogram of an image after preprocessing.
    Args:
        image (np.ndarray): Input grayscale image.
        num_buckets (int): Number of histogram bins.
    Returns:
        np.ndarray: Normalized histogram.
    """
    # Preprocess the image for scaling invariance
    processed_image = preprocess_image(image)

    # Compute histogram
    hist = calcHist([processed_image], [0], None, [num_buckets], [0, 256])
    return normalize(hist, hist).flatten()

def compare_images_histograms(image_paths, threshold):
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
    for img_path in tqdm(image_paths, desc="Loading Images", unit="image"):
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


    """
    Compare images using histogram similarity with dynamic threshold adjustment.
    Args:
        image_paths (list): List of image file paths.
        initial_threshold (float): Initial similarity threshold for considering duplicates.
        alpha (float): Scaling factor for adjusting the dynamic threshold.
    Returns:
        dict: Dictionary where each key is an image, and its value is a list of similar images.
    """
    similar_dict = {}
    similarity_scores = []

    # Track histogram computation and comparison
    histograms = []
    for img_path in tqdm(image_paths, desc="Loading Images", unit="image"):
        img = imread(img_path, IMREAD_GRAYSCALE)
        if img is None:
            continue
        hist = compute_histogram(img)
        histograms.append((img_path, hist))

    # Compare histograms and collect similarity scores
    for i in tqdm(range(len(histograms)), desc="Computing Similarity Scores", unit="image"):
        current_image, current_hist = histograms[i]
        for j in range(i + 1, len(histograms)):
            other_image, other_hist = histograms[j]
            similarity = 1 - compareHist(current_hist, other_hist, HISTCMP_BHATTACHARYYA)
            similarity_scores.append((current_image, other_image, similarity))

    # Calculate dynamic threshold based on mean and standard deviation
    if similarity_scores:
        mean_score = sum(score for _, _, score in similarity_scores) / len(similarity_scores)
        std_dev = (sum((score - mean_score) ** 2 for _, _, score in similarity_scores) / len(similarity_scores)) ** 0.5
        dynamic_threshold = mean_score + alpha * std_dev
        print(f"\nDynamic Threshold: {dynamic_threshold:.4f}")
    else:
        dynamic_threshold = initial_threshold

    # Compare histograms and build similarity dictionary
    for img1, img2, similarity in similarity_scores:
        if similarity > dynamic_threshold:
            if img1 not in similar_dict:
                similar_dict[img1] = []
            similar_dict[img1].append(img2)

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

def resolve_transitive_duplicates(similar_images):
    """
    Resolves transitive relationships in a dictionary of duplicate images.

    Args:
        similar_images (dict): Dictionary where keys are image filenames and values are lists of similar images.

    Returns:
        dict: Dictionary with transitive duplicates resolved.
    """
    # Create a mapping of connected components using a union-find approach
    from collections import defaultdict

    # Union-Find (Disjoint Set)
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Initialize the parent mapping
    for key, values in similar_images.items():
        if key not in parent:
            parent[key] = key
        for value in values:
            if value not in parent:
                parent[value] = value
            union(key, value)

    # Group all images by their root parent
    groups = defaultdict(set)
    for key in parent:
        root = find(key)
        groups[root].add(key)

    # Convert groups into the desired output format
    resolved_dict = {}
    for group in groups.values():
        group = list(group)
        key = group[0]
        resolved_dict[key] = sorted(group[1:])

    return resolved_dict

def main():
    # root = Tk()
    image_folder_path = select_image_folder()
    print("image_folder_path ", image_folder_path)
    chdir(image_folder_path)

    image_files = get_image_files()
    if image_files:
        print("Images found ", len(image_files))
    else:
        print("No image files found")
    
    similar_groups = compare_images_histograms(image_files, threshold=0.8)
    print("Similar ", similar_groups)
    similar_groups = resolve_transitive_duplicates(similar_groups)
    if similar_groups:
        print("Similar images ", similar_groups)
    else:
        print("No similar Images found!")


    move_duplicate_images(similar_groups)

if __name__ == "__main__":
    main()
    print("All done!")
    input("\nPress Enter to close this window...")
