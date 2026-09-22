import cv2
import numpy as np

def calculate_accuracy(output_path, image_path):
    try:
        # Load images
        output_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if output_image is None or ground_truth_image is None:
            raise ValueError("One of the images could not be loaded.")

        # Ensure both images are of the same size
        if output_image.shape != ground_truth_image.shape:
            raise ValueError("Output and ground truth images must have the same dimensions.")

        # Calculate accuracy
        correct_predictions = np.sum(output_image == ground_truth_image)
        total_pixels = output_image.size
        accuracy = (correct_predictions / total_pixels)*100

        return accuracy

    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return None