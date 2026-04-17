from PIL import Image
import os

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.load()
            img = img.convert("RGB")

            # remove tiny images
            if img.size[0] < 128 or img.size[1] < 128:
                return False

        return True
    except:
        return False


def clean_dataset(folder):
    bad_images = []

    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            if not is_valid_image(path):
                bad_images.append(path)

    print(f"\nFound {len(bad_images)} bad images\n")

    for path in bad_images:
        print("Removing:", path)
        os.remove(path)


if __name__ == "__main__":
    clean_dataset("data/train")
    clean_dataset("data/val")
