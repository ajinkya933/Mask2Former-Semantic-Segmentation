import json
import torch
from PIL import Image
import requests
import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms  
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation






def visualize_segmentation_map(segmentation_map, color_palette, filename):
    # Create an empty image with the same size as the segmentation map
    h, w = segmentation_map.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each class ID to its corresponding RGB color
    for class_id, color in enumerate(color_palette):
        class_mask = (segmentation_map == class_id)
        image[class_mask] = color

    # Filename 
    output_folder = 'segmented-output'
    
    # Saving the image 
    cv2.imwrite(output_folder+'/'+filename, image) 















# Load labels
f = open('sidewalk-semantic/id2label.json') 
   
id2label = json.load(f)
id2label = {int(k):v for k,v in id2label.items()}
print(id2label)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Grab the trained model and processor
model = Mask2FormerForUniversalSegmentation.from_pretrained("model", ignore_mismatched_sizes=True).to(device)
processor = AutoImageProcessor.from_pretrained("model", ignore_mismatched_sizes=True)


import os

# Define the directory path
directory_path = "image-input"

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg"):
        # Perform actions on the .asm files
        print(f"Processing {directory_path+'/'+filename}")

        image = Image.open(directory_path+'/'+filename)
        # Normalize the input image and convert it to float32
        transform = transforms.Compose([transforms.ToTensor()])
        normalized_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(normalized_input)

        # you can pass them to processor for postprocessing
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(model.config.id2label))]
        predicted_map_cpu = predicted_map.cpu().numpy()
        visualize_segmentation_map(predicted_map_cpu, color_palette, filename)
    else:
        continue


