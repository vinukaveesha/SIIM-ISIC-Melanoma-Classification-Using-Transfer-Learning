import torchvision.models as models

# Load the InceptionV3 model with pretrained weights
model = models.inception_v3(pretrained=True)

print(model)
