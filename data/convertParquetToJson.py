import pandas as pd

def convert_parquet_to_json(parquet_file, json_file):
    # Read Parquet file
    df = pd.read_parquet(parquet_file)
    
    # Convert DataFrame to JSON with proper formatting
    df.to_json(json_file, orient="records", indent=4)  # Ensures correct JSON format
    
    print(f"Conversion complete: {json_file} created successfully.")

# Convert test, train, and validation Parquet files to properly formatted JSON
convert_parquet_to_json("test-00000-of-00001.parquet", "test.json")
convert_parquet_to_json("train-00000-of-00001.parquet", "train.json")
convert_parquet_to_json("validation-00000-of-00001.parquet", "validation.json")
