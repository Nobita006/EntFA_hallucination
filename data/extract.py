import json

# Load test.json
with open('data/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write the source field to test.source
with open('data/test.source', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(item['source'] + '\n')
