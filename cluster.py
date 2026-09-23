import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

#load embedded vectors
features = np.load("features.npy")
print("embedding shape:",features.shape)

# load postcard image path
image_folder = "static/Images"
image_path = sorted([
    os.path.join(image_folder, file)
    for file in os.listdir(image_folder)
    if file.endswith(".jpg")
])
print("total images:",len(image_path))

# sanity check
if len(image_path)!= len(features):
    print("Wrong:")
    print("image length and embedding length don't match")
    print("images:", len(image_path))
    print("embeddings:", len(features))


# group similar postcards
cluster_num=30
kmeans_model = KMeans(
    n_clusters=cluster_num,
    random_state=42,
    n_init=10
)
cluster_labels = kmeans_model.fit_predict(features)
# save cluster labels
np.save("cluster_labels.npy", cluster_labels)
print(" cluster labels")

# print cluster size


for idx in range(cluster_num):
    total = len(cluster_labels[cluster_labels==idx])
    print("cluster",idx,"has",total,"samples")

# inspect one cluster visually
cluster_id=0
indexes=np.where(cluster_labels==cluster_id)[0]
print("\nShowing postcards from cluster:", cluster_id)
plt.figure(figsize=(12,8))
for i, idx in enumerate(indexes[:9]):
    # open postcard image
    img= Image.open(image_path[idx])
    img=img.resize((200,200))
    plt.subplot(3,3,i+1)
    plt.imshow(img)

    plt.title(f"img {idx}")
    plt.axis("off")
plt.tight_layout()
plt.show()                                                  

# similar postcard search
q_id = 50
q_embedding = features[q_id]
similarities = cosine_similarity(
    [q_embedding],
    features

)[0]

simi_index = similarities.argsort()[::-1]
print("\nsimilar postcards:\n")
plt.figure(figsize=(14,4))

for i, idx in enumerate(simi_index[:5]):
   img = Image.open(image_path[idx])
   img = img.resize((200,200)) 
   plt.subplot(1,5,i+1)
   plt.imshow(img)

   score = similarities[idx]
   plt.title(f"{score:.2f}")
   plt.axis("off")
plt.tight_layout()
plt.show()




#dimensionally reduction, 2D visualization
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="random"
)

sample_size = 2000
sample_features = features[:sample_size]
sample_labels = cluster_labels[:sample_size]
points_2d = tsne.fit_transform(sample_features)

# plotting

plt.figure(figsize=(10,8))
x_points=points_2d[:,0]
y_points=points_2d[:,1]
scatter = plt.scatter(
x_points,
y_points,
c=sample_labels,
alpha=0.8
)
# label cluster centers
for i in range(cluster_num):
    center_x = x_points[sample_labels==i].mean()
    center_y = y_points[sample_labels==i].mean()
    plt.text(center_x,center_y,str(i), fontsize=12)

plt.title("Postcard Clusters - (15 Groups)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.colorbar(scatter)
plt.tight_layout()

plt.show()
