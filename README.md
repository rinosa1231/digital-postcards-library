# Digital Postcards Library

A Flask-based digital postcard library that combines computer vision, semantic search, clustering, and geospatial visualization to explore a large collection of historical postcards.

## Project Overview

This project provides an interactive web application for exploring approximately 10,600 digital postcard images together with their metadata.

The application uses OpenAI CLIP to create image embeddings and enables users to search and explore postcards based on visual and semantic similarity.

## Key Features

-  Semantic image search using CLIP
-  Country-based filtering
-  Geographic visualization of postcard locations
-  Postcard journey and location exploration
-  Image clustering using K-Means
-  Cluster visualization using t-SNE
-  Topic evolution over time
-  Interactive postcard browsing
-  Geographical distance-based exploration

## Technologies Used

- Python
- Flask
- PyTorch
- OpenAI CLIP (ViT-B/32)
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Plotly
- Folium
- Geopy
- Pillow
- HTML/CSS

## Machine Learning Approach

The project uses CLIP to transform postcard images into numerical feature representations.

These embeddings are then used for:

1. Semantic similarity search
2. K-Means clustering
3. Cosine similarity analysis
4. t-SNE visualization

The clustering component groups visually and semantically similar postcards to support exploration of the collection.

## Application Structure

```text
digital-postcards-library/
│
├── main.py
├── search.py
├── cluster.py
├── image_features.py
├── gen_coords.py
├── generate_country_centers.py
│
├── data.json
├── location_coords.json
├── country_centers.json
│
├── static/
│   └── Images/
│
├── templates/
│   ├── index.html
│   ├── clusters.html
│   ├── map.html
│   └── welcome.html
│
├── README.md
├── Requirements.txt
└── .gitignore

## Dataset
The original project contains approximately 10,600 postcard images and associated metadata.

The postcard image collection and large generated feature files are not included in this public GitHub repository because of their size and dataset/distribution considerations.

The application therefore requires the original project data files to be available locally.

## Project Background
This project was independently designed and developed as part of an academic project.

The project requirements were provided by the university, while the implementation, machine learning workflow, data processing, web application, and interactive visualizations were developed by me.

The project has now been reorganized and documented as a portfolio project to demonstrate practical experience with Python, machine learning, computer vision, semantic search, clustering, and geospatial data visualization.

## Author
**Meera Sahib Fathima Rinosa**
MSc Computational Social Systems
TU Graz