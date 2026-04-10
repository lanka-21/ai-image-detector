from PIL import Image
import os

root_dir = r"L:\PYTHONNNN\projects\camera\data"

valid_extensions = (".jpg", ".jpeg", ".png")

def verify_images(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(subdir, file)

            # 🔴 STEP 1: Remove non-image files
            if not file.lower().endswith(valid_extensions):
                print(f"Removing non-image: {path}")
                os.remove(path)
                continue  # skip further checks

            # 🟢 STEP 2: Verify image
            try:
                img = Image.open(path)
                img.verify()
            except:
                print(f"Removing corrupted: {path}")
                os.remove(path)

verify_images(root_dir)