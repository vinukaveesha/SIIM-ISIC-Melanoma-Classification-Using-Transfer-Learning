import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def apply_filter_dicom(dicom_file_path, filter_type="gaussian", kernel_size=3, sigma=1):
    """
    Applies a chosen filter to a DICOM image.

    Parameters:
        dicom_file_path (str): Path to the DICOM file.
        filter_type (str): Type of filter to apply: 'median', 'gaussian', or 'bilateral'.
        kernel_size (int): Size of the kernel for median and Gaussian filters.
        sigma (float): Sigma value for Gaussian and Bilateral filters.

    Returns:
        numpy.ndarray: Processed image.
    """
    # Load the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = dicom_data.pixel_array.astype(np.float32)

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

    else:
        raise ValueError(
            "Invalid filter_type. Choose from 'median', 'gaussian', or 'bilateral'."
        )

    return processed_image
