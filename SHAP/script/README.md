# Shap Analysis Script

## Overview
This Python script performs Shapley value analysis on the set of images using TensorFlow, Shap, and other libraries. The goal is to identify clusters of pixels that collaboratively contribute to a neural network's identification of an image, specifically in the context of tumor detection.

## Dependencies
Make sure you have the following dependencies installed:

- TensorFlow
- Shap
- OpenCV (cv2)
- NumPy
- Matplotlib
- psutil

## Usage

The script is executed from the command line and requires several command-line arguments:

- `/path/to/results`: Directory where the results will be saved.
- `/path/to/explanations`: Directory where Shap explanations will be stored.
- `/path/to/model`: Path to the saved TensorFlow model.
- `/path/to/imagesr_path`: Path where all the images are stored.
- `number_of_cores`: Number of CPU cores to be used for parallel processing.

### Example usage:

```bash
python3 shap_script.py /path/to/results /path/to/explanations /path/to/model /path/to/images 4
```