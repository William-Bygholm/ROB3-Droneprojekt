import json

path = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n=== Annotations (første 5) ===")
for ann in data["annotations"][:5]:
    print(json.dumps(ann, indent=4))

print("\n=== Categories ===")
print(json.dumps(data["categories"], indent=4))
