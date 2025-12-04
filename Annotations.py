import json

# Sti til din JSON-fil
JSON_FILE = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"

# Indlæs JSON
with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Tæl annotations
num_annotations = len(data.get("annotations", []))

print(f"Der er i alt {num_annotations} annotations i JSON-filen.")
