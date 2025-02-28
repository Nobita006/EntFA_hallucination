import pandas as pd

# Load the CSV file
csv_path = "abstract_highlights_filtered.csv"
data = pd.read_csv(csv_path)

# Rename columns for clarity
data.columns = ["source", "hypothesis"]

# Clean the 'hypothesis' column (remove @highlight and join sentences)
def clean_highlights(highlights):
    sentences = highlights.split("@highlight")[1:]  # Split and ignore the first empty entry
    return " ".join([sent.strip() for sent in sentences if sent.strip()])

data["hypothesis"] = data["hypothesis"].apply(clean_highlights)

# Save the processed data into the required text files
source_path = "data/test.source"
hypothesis_path = "data/test.hypothesis"

data["source"].to_csv(source_path, index=False, header=False)
data["hypothesis"].to_csv(hypothesis_path, index=False, header=False)

print(f"Source saved to: {source_path}")
print(f"Hypothesis saved to: {hypothesis_path}")
