import os

# Sti til mappen med billeder
folder_path = r"C:\Users\alexa\Desktop\Pos"

# Navn alle billeder starter med
prefix = "pos_3 mili 2 onde"   # f.eks. "IMG_"

# Gå gennem alle filer i mappen
for filename in os.listdir(folder_path):
    if filename.startswith(prefix):
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            print(f"Slettet: {file_path}")
        except Exception as e:
            print(f"Kunne ikke slette {file_path}: {e}")
