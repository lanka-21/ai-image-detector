import os
import shutil

# =========================
# PATHS
# =========================

source_folder = r'E:\realcocodataset'
target_base = r'E:\dataset\real'

# =========================
# CATEGORY GROUPS
# =========================

person_keywords = ['person']

animal_keywords = [
    'dog', 'cat', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'bird'
]

fruit_keywords = ['apple', 'banana', 'orange']

# =========================
# CREATE FOLDERS
# =========================

folders = ['person', 'animal', 'fruits', 'others']

for folder in folders:
    os.makedirs(os.path.join(target_base, folder), exist_ok=True)

# =========================
# MOVE FILES
# =========================

print("🔄 Organizing real dataset...")

for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    if not os.path.isfile(file_path):
        continue

    name = filename.lower()

    moved = False

    # PERSON
    for key in person_keywords:
        if key in name:
            shutil.move(file_path, os.path.join(target_base, 'person', filename))
            moved = True
            break

    # ANIMALS
    if not moved:
        for key in animal_keywords:
            if key in name:
                shutil.move(file_path, os.path.join(target_base, 'animal', filename))
                moved = True
                break

    # FRUITS
    if not moved:
        for key in fruit_keywords:
            if key in name:
                shutil.move(file_path, os.path.join(target_base, 'fruits', filename))
                moved = True
                break

    # OTHERS (fallback)
    if not moved:
        shutil.move(file_path, os.path.join(target_base, 'others', filename))

print("✅ DONE — Real dataset organized!")
