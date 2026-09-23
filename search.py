import numpy as np
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
#Load saved features
features = np.load("features.npy")
names = np.load("names.npy")
#Convert to tensor
features = torch.tensor(features).to(device)
#search what we want
query = "beach"
query="city"
query="vehicles"
query="nature"
query = "paintings"
query = "mountain"
query = "flowers"
query = "river"
query = "architecture"
query = "animals"
query = "birds"
query = "letter"

with torch.no_grad():
    text = clip.tokenize([query]).to(device)
    text_features = model.encode_text(text)
#compare images with the text   
similarity = (features @ text_features.T).squeeze()
#top 10 matches
top_k = 10
indices = similarity.topk(top_k).indices
print("\nTop results:")
for i in indices:
    print(names[i])