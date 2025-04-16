import matplotlib.pyplot as plt
import numpy as np
import os
from sentence_transformers import util
from vendor_matcher import VendorMatcher

# Get absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define output directory for saving threshold selection plots
output_dir = os.path.join(current_dir, "threshold_selection_plots")

# Create the output directory if it doesn't exist; clear existing plots if it does
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Initialize VendorMatcher with the dataset path
matcher = VendorMatcher(os.path.join(current_dir, "data", "G2 software - CRM Category Product Overviews.csv"))

# Define feature-based queries (used to test capacity/feature similarity)
feature_queries = [
    "Eye-tracking for customer behavior",
    "Emotion Recognition",
    "Customization",
    "Pipeline Management",
    "Desktop Integration",
    "Budgeting"
]

# Define category-based queries (used to test category similarity)
category_queries = [
    "CRM",
    "ERP",
    "Accounting & Finance Software"
]

# Define similarity thresholds to test (from 0.0 to 1.0, 51 points)
thresholds = np.linspace(0, 1, 51)

# Generate and save plots for each feature query
for query in feature_queries:
    # Encode the input query using the same embedding model
    query_embedding = matcher.model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between query and all software feature embeddings
    similarity_scores = util.cos_sim(query_embedding, matcher.feature_embeddings_feature)[0].cpu().numpy()

    # Count how many software entries match at each threshold
    num_matches = [(similarity_scores >= t).sum() for t in thresholds]

    # Plot and save the result
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, num_matches, marker='o')
    plt.title(f'Matches vs Threshold for Feature: \"{query}\"')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Matching Software')
    plt.grid(True)
    plt.tight_layout()

    filename = 'Feature_' + query.lower().replace(" ", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

# Generate and save plots for each category query
for query in category_queries:
    # Encode category query and compare to all category embeddings
    query_embedding = matcher.model.encode(query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, matcher.feature_embeddings_category)[0].cpu().numpy()

    # Count how many software entries match at each threshold
    num_matches = [(similarity_scores >= t).sum() for t in thresholds]

    # Plot and save the result
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, num_matches, marker='o')
    plt.title(f'Matches vs Threshold for Category: \"{query}\"')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Matching Software')
    plt.grid(True)
    plt.tight_layout()

    filename = 'Category_' + query.lower().replace(" ", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
