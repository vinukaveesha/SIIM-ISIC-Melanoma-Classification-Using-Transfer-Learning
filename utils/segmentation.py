import cv2
import numpy as np
from skimage import morphology

def segment_lesion(image, method="threshold", threshold=0.5):
    """
    Segments the lesion area in an image.
    
    Parameters:
        image (numpy array): Input grayscale or RGB image.
        method (str): Segmentation method. Options are "threshold" or "contour".
        threshold (float): Threshold value for segmentation, used if method="threshold".
    
    Returns:
        segmented_image (numpy array): Segmented binary mask of the lesion.
    """
    if method == "threshold":
        # Convert image to grayscale if it's in RGB format
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply global thresholding
        _, binary_mask = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Remove small artifacts and fill holes
        binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=100)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=100)
        
    elif method == "contour":
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use adaptive thresholding and find contours
        binary_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for the largest contour
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        binary_mask = mask

    else:
        raise ValueError("Unknown segmentation method. Use 'threshold' or 'contour'.")

    return binary_mask.astype(np.uint8)

def apply_segmentation(image, method="threshold", threshold=0.5):
    """
    Applies segmentation and returns an image where the background is masked out.
    
    Parameters:
        image (numpy array): Original image.
        method (str): Segmentation method to apply.
        threshold (float): Threshold for segmentation, if applicable.
    
    Returns:
        segmented_image (numpy array): Original image with only the lesion region.
    """
    mask = segment_lesion(image, method=method, threshold=threshold)
    return cv2.bitwise_and(image, image, mask=mask)
