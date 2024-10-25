import pydicom
import matplotlib.pyplot as plt
from preprocessing import apply_filter_dicom

# Load the DICOM file
dcm_file_path = "data/train/ISIC_0015719.dcm"
dicom_data = pydicom.dcmread(dcm_file_path)

# Print metadata
# print(dicom_data)


def view_dicom_file(dicom_file_path=None, pixel_array=None):
    if dicom_file_path is None:
        # Access the pixel data
        pixel_array = dicom_data.pixel_array
    else:
        # Load the DICOM file
        dicom_data = pydicom.dcmread(dicom_file_path)

        # Access the pixel data
        pixel_array = dicom_data.pixel_array

    # Display the image
    plt.imshow(pixel_array, cmap="gray")
    plt.title("DICOM Image")
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.show()
