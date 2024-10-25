# from utils.data_viewer import view_dicom_file
from utils.preprocessing import apply_filter_dicom
import numpy as np
import pydicom

dcm_file_path = "data/train/ISIC_0015719.dcm"

dicom_data = pydicom.dcmread(dcm_file_path)
image = dicom_data.pixel_array.astype(np.float32)
print(image.shape)
