import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology

def watershed_segmentation(image):
    # Apply watershed segmentation to the denoised image
    markers = morphology.label(image > image.mean())
    segmentation_map = watershed(-image, markers, mask=image > image.mean())
    return segmentation_map
