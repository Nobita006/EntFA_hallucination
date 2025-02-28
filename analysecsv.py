import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import json

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# File paths
csv_path = "abstract_highlights_filtered.csv"
source_path = "data/test.source"
hypothesis_path = "data/test.hypothesis"

# Helper function to analyze entities
def analyze_entities(texts):
    entity_lengths = []
    start_positions = []
    end_positions = []

    for text in tqdm(texts, desc="Analyzing Entities"):
        doc = nlp(text)
        for ent in doc.ents:
            entity_lengths.append(len(ent.text))
            start_positions.append(ent.start_char)
            end_positions.append(ent.end_char)
    
    analysis = {
        "Average length": np.mean(entity_lengths),
        "Max length": np.max(entity_lengths),
        "Min length": np.min(entity_lengths),
        "Start Position": {"Max": np.max(start_positions), "Min": np.min(start_positions)},
        "End Position": {"Max": np.max(end_positions), "Min": np.min(end_positions)},
    }
    return analysis

# Function to print results nicely
def print_analysis(title, analysis):
    print(f"\n{title} Entity Length Analysis")
    print("Average Length:", analysis["Average length"])
    print("Max Length:", analysis["Max length"])
    print("Min Length:", analysis["Min length"])
    print("Start Position - Max:", analysis["Start Position"]["Max"], "Min:", analysis["Start Position"]["Min"])
    print("End Position - Max:", analysis["End Position"]["Max"], "Min:", analysis["End Position"]["Min"])

# Step 2: Load Data
print("Loading CSV file...")
csv_data = pd.read_csv(csv_path)
source_texts = csv_data["Abstract"].tolist()
hypothesis_texts = csv_data["Highlights"].apply(lambda x: " ".join(x.split("@highlight")[1:])).tolist()

print("Loading test.source and test.hypothesis...")
with open(source_path, "r") as f:
    test_source = [line.strip() for line in f.readlines()]

with open(hypothesis_path, "r") as f:
    test_hypothesis = [line.strip() for line in f.readlines()]

# Step 3: Analyze Entities
print("\nAnalyzing abstract_highlights_filtered.csv...")
abstract_analysis = analyze_entities(source_texts)
highlight_analysis = analyze_entities(hypothesis_texts)

print("\nAnalyzing test.source...")
test_source_analysis = analyze_entities(test_source)

print("\nAnalyzing test.hypothesis...")
test_hypothesis_analysis = analyze_entities(test_hypothesis)

# Step 4: Print Results
print_analysis("abstract_highlights_filtered.csv - Abstracts", abstract_analysis)
print_analysis("abstract_highlights_filtered.csv - Highlights", highlight_analysis)
print_analysis("test.source", test_source_analysis)
print_analysis("test.hypothesis", test_hypothesis_analysis)
