import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def load_dicom_image(file_path):
    dicom_image = pydicom.dcmread(file_path)
    image_array = dicom_image.pixel_array

    # Normalize to the range [0, 255]
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
        image_array = image_array.astype(np.uint8)

    return image_array


def dicom_to_gray_scale(image):

    # Convert the image to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize the image to 200x200 pixels
    image = cv2.resize(image, (200, 200))

    return image


def apply_ahe(image):
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the input image
    enhanced_image = clahe.apply(image)

    return enhanced_image


def hair_remove(image, grayScale):

    image = cv2.resize(image, (200, 200))

    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

    # Convert the inpainted image back to grayscale
    final_gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    return final_gray_image


def weighted_addition_enhance_image(
    image, alpha=4, beta=-4, gamma=128, blur_kernel_size=(3, 3), sigma=256 / 10
):
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Input image is not valid.")

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, blur_kernel_size, sigma)

    # Apply the weighted addition
    enhanced_image = cv2.addWeighted(image, alpha, blurred_image, beta, gamma)

    return enhanced_image


def apply_filter_dicom(
    image,
    filter_type="gaussian",
    kernel_size=1,
    sigma=1,
    noise_level=10,
    template_window_size=10,
    search_window_size=21,
):
    if filter_type == "median":
        # Median filter in the spatial domain
        processed_image = cv2.medianBlur(image, kernel_size)

    elif filter_type == "gaussian":
        # Gaussian filter in frequency domain
        dft = fftshift(fft2(image))
        # Create Gaussian mask in frequency domain
        rows, cols = image.shape
        x = np.linspace(-cols // 2, cols // 2, cols)
        y = np.linspace(-rows // 2, rows // 2, rows)
        x, y = np.meshgrid(x, y)
        gaussian_filter = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
        dft_filtered = dft * gaussian_filter
        processed_image = np.abs(ifft2(ifftshift(dft_filtered)))

    elif filter_type == "bilateral":
        # Bilateral filter in spatial domain
        processed_image = cv2.bilateralFilter(
            image, d=kernel_size, sigmaColor=sigma, sigmaSpace=sigma
        )

    elif filter_type == "nlm":
        h = noise_level  # level of noise
        templateWindowSize = template_window_size
        searchWindowSize = search_window_size

        # Apply the Non-Local Means filter
        processed_image = cv2.fastNlMeansDenoising(
            image, None, h, templateWindowSize, searchWindowSize
        )

    else:
        raise ValueError(
            "Invalid filter_type. Choose from 'median', 'gaussian', or 'bilateral'."
        )

    return processed_image


def apply_canny_edge_detection(image, low_threshold, high_threshold):

    # Apply Canny edge detector
    edges = cv2.Canny(image, low_threshold, high_threshold)

    return edges


def watershed_segmentation(gray_image):
    # Convert the grayscale image to RGB since watershed needs a 3-channel image
    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Convert to binary image via thresholding
    _, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Ensure markers are of type int32
    markers = markers.astype(np.int32)

    # Apply watershed
    cv2.watershed(image, markers)

    # Color only inner borders
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if markers[i, j] == -1 and (
                i > 0 and i < image.shape[0] - 1 and j > 0 and j < image.shape[1] - 1
            ):
                image[i, j] = [255, 0, 0]  # Red color for inner borders

    return image


def watershed_segmentation_without_denoising_implicitly(gray_image):
    # Convert the grayscale image to RGB and make a copy since watershed needs a 3-channel image
    original_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    image = original_image.copy()

    # Convert to binary image via thresholding
    _, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise removal (optional)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Ensure markers are of type int32
    markers = markers.astype(np.int32)

    # Apply watershed
    cv2.watershed(image, markers)

    # Color only inner borders, excluding the edge of the image
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if markers[i, j] == -1 and (
                i > 0 and i < image.shape[0] - 1 and j > 0 and j < image.shape[1] - 1
            ):
                image[i, j] = [255, 0, 0]  # Red color for inner borders

    return image


def watershed_segmentation_full_color(gray_image):

    # Convert the grayscale image to RGB since watershed needs a 3-channel image
    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Convert to binary image via thresholding
    _, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise removal (optional)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Ensure markers are of type int32
    markers = markers.astype(np.int32)

    # Apply watershed
    cv2.watershed(image, markers)

    # Create a random color map
    segment_colors = np.random.randint(0, 255, size=(markers.max() + 1, 3))

    # Color each segment
    segmented_image = segment_colors[markers]

    return segmented_image.astype(np.uint8)


def watershed_segmentation_black_white(gray_image):
    # Convert the grayscale image to RGB since watershed needs a 3-channel image
    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Convert to binary image via thresholding
    _, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noise removal (optional)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Ensure markers are of type int32
    markers = markers.astype(np.int32)

    # Apply watershed
    cv2.watershed(image, markers)

    # All segment areas as black
    black_white_image = np.where(markers > 1, 0, 225).astype(np.uint8)

    return black_white_image


# Example usage
dicom_file_path = "data/train/ISIC_0052212.dcm"
original_image = load_dicom_image(dicom_file_path)
gray_sclaed = dicom_to_gray_scale(original_image)
hair_removed = hair_remove(image=original_image, grayScale=gray_sclaed)
enhanced_image = apply_ahe(hair_removed)
denoised = apply_filter_dicom(enhanced_image, filter_type="nlm")
edges_detected = apply_canny_edge_detection(denoised, 100, 200)
segmented = watershed_segmentation(denoised)

# Plotting the original and enhanced images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(gray_sclaed, cmap="gray")
plt.title("Gray scaled DICOM Image")
plt.axis("off")

# Enhanced image
plt.subplot(2, 2, 2)
plt.imshow(enhanced_image, cmap="gray")
plt.title("weighted addition enhanced Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(denoised, cmap="gray")
plt.title("Denoised Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(segmented, cmap="gray")
plt.title("Segamented Image")
plt.axis("off")


plt.show()
