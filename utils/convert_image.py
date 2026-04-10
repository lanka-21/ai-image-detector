import os
from PIL import Image

# root dataset path
root_dir = r"L:\PYTHONNNN\projects\camera\data"

valid_extensions = (".jpg", ".jpeg", ".png")

def convert_to_jpg(path):
    try:
        img = Image.open(path).convert("RGB")
        new_path = os.path.splitext(path)[0] + ".jpg"
        img.save(new_path, "JPEG", quality=95)
        os.remove(path)
        print(f"Converted: {path} -> {new_path}")
    except Exception as e:
        print(f"Failed: {path}, Error: {e}")

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)

        if not file.lower().endswith(valid_extensions):
            convert_to_jpg(file_path)