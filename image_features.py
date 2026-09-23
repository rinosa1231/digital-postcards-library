import os
import numpy as np
import torch
import clip
from PIL import Image

# Load model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "static/images"
all_features = []
all_names = []
for filename in os.listdir(image_folder):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(image_folder, filename)

        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(image)
                features /= features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy()[0])

            all_names.append(filename)

            print("Processed:", filename)
        except Exception as e:
            print("Error:", filename, e)
all_features = np.array(all_features)

np.save("features.npy", all_features)

np.save("names.npy", np.array(all_names))

print("DONE")



    




    

