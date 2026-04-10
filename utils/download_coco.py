import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

annFile = r'C:\Users\klank\Downloads\annotations_trainval2017\annotations\instances_train2017.json'
save_path = r'E:\realcocodataset'

# Strong + balanced categories
my_categories = [
    'person',
    'dog', 'cat', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',
    'bird',
    'apple', 'banana', 'orange'
]

MAX_IMAGES_PER_CATEGORY = 600  # balanced limit
MIN_SIZE = 500                 # quality filter

# =========================
# SETUP
# =========================

os.makedirs(save_path, exist_ok=True)

print("🔄 Loading COCO annotations...")
coco = COCO(annFile)

# =========================
# DOWNLOAD FUNCTION
# =========================

def download_images(category):
    print(f"\n📥 Processing category: {category}")

    catIds = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    count = 0

    for im in tqdm(images):
        if count >= MAX_IMAGES_PER_CATEGORY:
            break

        # Strong quality filter
        if im['width'] >= MIN_SIZE and im['height'] >= MIN_SIZE:
            try:
                url = im['coco_url']
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    filename = f"{category}_{im['id']}.jpg"
                    filepath = os.path.join(save_path, filename)

                    # Avoid duplicate download
                    if not os.path.exists(filepath):
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        count += 1

            except:
                continue

    
    print(f"✅ {category}: {count} images saved")

# =========================
# MAIN LOOP
# =========================

for cat in my_categories:
    download_images(cat)

print("\n🎉 DONE — DATASET READY")
print(f"📁 Saved at: {save_path}")