import pydicom

# Load the DICOM file
dcm_file_path = "data/train/ISIC_0015719.dcm"
dicom_data = pydicom.dcmread(dcm_file_path)

# Print metadata
print(dicom_data)
