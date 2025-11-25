import os 

# Sti til mappen med billederne
folder_path = r"C:\Users\alexa\Desktop\Neg"

# Mappings af danske bogstaver til normale
char_map = {
    'æ': 'e',
    'ø': 'o', 
    'å': 'a',
    'Æ': 'E',
    'Ø': 'O',
    'Å': 'A'
}

# Gennemgå alle filer i mappen
for filename in os.listdir(folder_path):
    new_name = filename
    for dan_char, repl_char in char_map.items():
        new_name = new_name.replace(dan_char, repl_char)
    
    if new_name != filename:
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

print("Færdig!")