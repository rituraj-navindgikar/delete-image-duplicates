# Duplica
This project solves the problem of having a large collection of images (100GB+ in my case) with duplicate files scattered across folders. The tool efficiently identifies duplicate images and helps you organize or remove them, saving both storage and time

The solution is based on *image histogram comparison*, which is a computationally efficient method for identifying visually similar images. Using OpenCV and other lightweight libraries, this approach can handle tens of thousands of images without breaking a sweat

---

## Features
- 🚀 Fast and Efficient: Uses histogram-based image comparison for speed, ideal for large datasets
- 📂 Handles Common Formats: Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, and I am trying for `HEIC/HEIF`
- 🖼 Accurate Duplicate Detection: Identifies duplicates based on a similarity threshold.
- 📁 Custom Folder Management: Automatically organizes duplicates into a separate folder for easy management.

---

## Getting Started
### My Preference 
I believe it's best to convert this in .exe file to run on any windows system without worrying about installing python and its dependencies
```bash
pyinstaller --onefile main.py
```
You can find the `.exe` file [here](https://drive.google.com/drive/folders/1dwrjuz5RSkdbU6Xnj_lR8b1uOE3ky19J?usp=sharing)

---
### Installation
1. Clone the repository:
```bash
git clone https://github.com/rituraj-navindgikar/delete-image-duplicates
cd delete-image-duplicates
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

--- 

### Usage
1. Run the script:
```bash
python main.py
```
2. Select the folder containing your images when prompted
3. The script will:

- Compare all images in the selected folder
- Identify duplicates based on a similarity threshold (default: 0.8)
- Move duplicate images into a folder named !duplicate_images

--- 

### How It Works
The tool uses a histogram-based method to compare images. This involves:
- Converting each image to grayscale
- Computing the histogram for pixel intensity distribution
- Comparing histograms using `Bhattacharyya distance` to measure similarity
- Grouping similar images together if their similarity exceeds a user-defined threshold
- For HEIC/HEIF images, the tool leverages the pyheif library to decode and process these formats seamlessly (under progress)


### Example Output
When the script runs, you’ll see:
- Progress updates for image processing.
- Similar groups of images displayed as:

```bash
Similar Images:
image1.jpg : image2.jpg, image3.jpg
image4.jpg : image5.jpg

Duplicate images moved to the !duplicate_images folder
```

### Requirements
Python 3.7 or higher
Libraries: OpenCV, TQDM, Pillow, pyheif

### Contributing
Feel free to open issues or submit pull requests to improve the tool. Suggestions and feedback are always welcome!

Rituraj Navindgikar

