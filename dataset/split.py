import os
import cv2
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

IMAGE_FOLDER = "dataset/images"
METADATA = "dataset/HAM10000_metadata.csv"

OUTPUT_DIR = "dataset"

IMG_SIZE = 224

df = pd.read_csv(METADATA)

# create image path
df["path"] = df["image_id"].apply(lambda x: os.path.join(IMAGE_FOLDER, x + ".jpg"))

#preprocess

def preprocess_image(src_path, dst_path):

    img = cv2.imread(src_path)

    if img is None:
        return

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    img = img / 255.0

    # convert back to 0-255 for saving
    img = (img * 255).astype(np.uint8)

    cv2.imwrite(dst_path, img)

#split dataset

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["dx"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["dx"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

#process image

def process_dataset(dataframe, dataset_type):

    for index, row in dataframe.iterrows():

        label = row["dx"]
        src = row["path"]

        dst_dir = os.path.join(OUTPUT_DIR, dataset_type, label)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, os.path.basename(src))

        preprocess_image(src, dst)

# process datasets
process_dataset(train_df, "train")
process_dataset(val_df, "val")
process_dataset(test_df, "test")

print("Dataset preprocessing and splitting completed!")