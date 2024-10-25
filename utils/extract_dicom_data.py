import os
import pydicom
import pandas as pd


# Function to extract metadata from a single DICOM file
def extract_dicom_metadata(dicom_path):
    # Load the DICOM file
    dicom_file = pydicom.dcmread(dicom_path)

    # Extract and format metadata as a list of lists
    metadata = []

    # File Meta Information Group
    metadata.append(["File Meta Information", ""])
    metadata.append(
        [
            "File Meta Information Group Length",
            dicom_file.file_meta.get((0x0002, 0x0000), "N/A"),
        ]
    )
    metadata.append(
        [
            "File Meta Information Version",
            dicom_file.file_meta.get((0x0002, 0x0001), "N/A"),
        ]
    )
    metadata.append(
        [
            "Media Storage SOP Class UID",
            dicom_file.file_meta.get((0x0002, 0x0002), "N/A"),
        ]
    )
    metadata.append(
        [
            "Media Storage SOP Instance UID",
            dicom_file.file_meta.get((0x0002, 0x0003), "N/A"),
        ]
    )
    metadata.append(
        ["Transfer Syntax UID", dicom_file.file_meta.get((0x0002, 0x0010), "N/A")]
    )
    metadata.append(
        ["Implementation Class UID", dicom_file.file_meta.get((0x0002, 0x0012), "N/A")]
    )
    metadata.append(
        [
            "Implementation Version Name",
            dicom_file.file_meta.get((0x0002, 0x0013), "N/A"),
        ]
    )
    metadata.append(
        [
            "Source Application Entity Title",
            dicom_file.file_meta.get((0x0002, 0x0016), "N/A"),
        ]
    )

    # General Image Information
    metadata.append(["General Image Information", ""])
    metadata.append(["Image Type", dicom_file.get((0x0008, 0x0008), "N/A")])
    metadata.append(["Instance Creator UID", dicom_file.get((0x0008, 0x0014), "N/A")])
    metadata.append(["SOP Class UID", dicom_file.get((0x0008, 0x0016), "N/A")])
    metadata.append(["SOP Instance UID", dicom_file.get((0x0008, 0x0018), "N/A")])
    metadata.append(["Study Date", dicom_file.get((0x0008, 0x0020), "N/A")])
    metadata.append(["Content Date", dicom_file.get((0x0008, 0x0023), "N/A")])
    metadata.append(["Study Time", dicom_file.get((0x0008, 0x0030), "N/A")])
    metadata.append(["Content Time", dicom_file.get((0x0008, 0x0033), "N/A")])
    metadata.append(["Accession Number", dicom_file.get((0x0008, 0x0050), "N/A")])
    metadata.append(["Modality", dicom_file.get((0x0008, 0x0060), "N/A")])
    metadata.append(["Manufacturer", dicom_file.get((0x0008, 0x0070), "N/A")])
    metadata.append(["Institution Name", dicom_file.get((0x0008, 0x0080), "N/A")])
    metadata.append(
        ["Referring Physician's Name", dicom_file.get((0x0008, 0x0090), "N/A")]
    )
    metadata.append(["Study Description", dicom_file.get((0x0008, 0x1030), "N/A")])

    # Patient Information
    metadata.append(["Patient Information", ""])
    metadata.append(["Patient's Name", dicom_file.get((0x0010, 0x0010), "N/A")])
    metadata.append(["Patient ID", dicom_file.get((0x0010, 0x0020), "N/A")])
    metadata.append(["Patient's Birth Date", dicom_file.get((0x0010, 0x0030), "N/A")])
    metadata.append(["Patient's Sex", dicom_file.get((0x0010, 0x0040), "N/A")])
    metadata.append(["Patient's Age", dicom_file.get((0x0010, 0x1010), "N/A")])

    # Anatomical and Study Details
    metadata.append(["Anatomical and Study Details", ""])
    metadata.append(["Body Part Examined", dicom_file.get((0x0018, 0x0015), "N/A")])
    metadata.append(["Study Instance UID", dicom_file.get((0x0020, 0x000D), "N/A")])
    metadata.append(["Series Instance UID", dicom_file.get((0x0020, 0x000E), "N/A")])
    metadata.append(["Study ID", dicom_file.get((0x0020, 0x0010), "N/A")])
    metadata.append(["Series Number", dicom_file.get((0x0020, 0x0011), "N/A")])
    metadata.append(["Instance Number", dicom_file.get((0x0020, 0x0013), "N/A")])
    metadata.append(["Patient Orientation", dicom_file.get((0x0020, 0x0020), "N/A")])

    # Image Specifications
    metadata.append(["Image Specifications", ""])
    metadata.append(["Samples per Pixel", dicom_file.get((0x0028, 0x0002), "N/A")])
    metadata.append(
        ["Photometric Interpretation", dicom_file.get((0x0028, 0x0004), "N/A")]
    )
    metadata.append(["Planar Configuration", dicom_file.get((0x0028, 0x0006), "N/A")])
    metadata.append(["Rows", dicom_file.get((0x0028, 0x0010), "N/A")])
    metadata.append(["Columns", dicom_file.get((0x0028, 0x0011), "N/A")])
    metadata.append(["Bits Allocated", dicom_file.get((0x0028, 0x0100), "N/A")])
    metadata.append(["Bits Stored", dicom_file.get((0x0028, 0x0101), "N/A")])
    metadata.append(["High Bit", dicom_file.get((0x0028, 0x0102), "N/A")])
    metadata.append(["Pixel Representation", dicom_file.get((0x0028, 0x0103), "N/A")])
    metadata.append(["Burned In Annotation", dicom_file.get((0x0028, 0x0301), "N/A")])
    metadata.append(
        ["Lossy Image Compression", dicom_file.get((0x0028, 0x2110), "N/A")]
    )

    # Pixel Data
    if hasattr(dicom_file, "pixel_array"):
        pixel_array_size = dicom_file.pixel_array.size
        metadata.append(["Pixel Data Size", pixel_array_size])
    else:
        metadata.append(["Pixel Data", "Not available"])

    return metadata


# Function to save metadata to an Excel file
def save_metadata_to_excel(metadata, output_file):
    # Convert the metadata list into a DataFrame
    df = pd.DataFrame(metadata, columns=["Tag", "Value"])

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Metadata saved to {output_file}")


# Iterate through all DICOM files in data/train and save metadata
def process_all_dicom_files(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".dcm"):  # Only process .dcm files
            dicom_path = os.path.join(input_directory, filename)
            output_file = os.path.join(
                output_directory, f"dicom_metadata_{filename}.xlsx"
            )

            # Extract metadata and save to Excel
            metadata = extract_dicom_metadata(dicom_path)
            save_metadata_to_excel(metadata, output_file)


# Example usage
input_directory = "data/train"
output_directory = "data/metadata"
process_all_dicom_files(input_directory, output_directory)
