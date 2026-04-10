import os
import argparse
import urllib.request
import zipfile

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index", type=int, default=1)
parser.add_argument("-r", "--range", type=int, default=None)
parser.add_argument("-o", "--output", type=str, default="images")

args = parser.parse_args()

index = args.index
range_max = args.range
output = args.output


# -----------------------------
# DOWNLOAD FUNCTION
# -----------------------------
def download(part_index):
    base_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images"
    
    part_name = f"part-{part_index:06}.zip"
    url = f"{base_url}/{part_name}"

    os.makedirs(output, exist_ok=True)

    zip_path = os.path.join(output, part_name)

    print(f"\nDownloading {part_name} ...")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    except Exception as e:
        print("Error downloading:", e)
        return

    # -----------------------------
    # EXTRACT
    # -----------------------------
    extract_path = os.path.join(output, f"part-{part_index:06}")
    os.makedirs(extract_path, exist_ok=True)

    print("Extracting...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extracted successfully!")
    except Exception as e:
        print("Extraction error:", e)


# -----------------------------
# MAIN
# -----------------------------
def main():
    if range_max:
        for i in range(index, index + range_max):
            download(i)
    else:
        download(index)


if __name__ == "__main__":
    main()
