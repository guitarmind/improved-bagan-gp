# %% --------------------------------------- Load Packages ---------------
import os
import numpy as np
from PIL import Image
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% --------------------------------------- Data Prep -------------------
# # Download data
# if "train" not in os.listdir():
#     os.system("cd ~/Capstone")
#     os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
#     os.system("unzip train.zip")

# # Read data
# DIR = 'train/'

# train = [f for f in os.listdir(DIR)]
# train_sorted = sorted(train, key=lambda x: int(x[5:-4]))
# imgs = []
# texts = []
# resize_to = 64
# for f in train_sorted:
#     if f[-3:] == 'png':
#         imgs.append(cv2.resize(cv2.imread(DIR + f), (resize_to, resize_to)))
#     else:
#         texts.append(open(DIR + f).read())

# imgs = np.array(imgs)
# texts = np.array(texts)

# le = LabelEncoder()
# le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
# labels = le.transform(texts)


dataset_folder = "/workspace/Kaggle/KolektorSDD"
target_size = 128

# https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in


def crop_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax + 1, xmin:xmax + 1], (ymin, ymax, xmin, xmax)

dataset_inputs = {}
dataset_labels = {}
defective_count = 0
for folder in os.listdir(dataset_folder):
    for filename in os.listdir(f"{dataset_folder}/{folder}"):

        if ".bmp" in filename:
            part_id = filename.split("_")[0]

            label_img = Image.open(f"{dataset_folder}/{folder}/{filename}")
            label_img = np.array(label_img)

            if np.sum(label_img) > 0:
                defective_count += 1

                # print(f"{folder}-{part_id} has defect:",
                # np.sum(label_img) > 0)

                dataset_labels[f"{folder}-{part_id}"] = label_img

        else:
            part_id = filename.split(".")[0]

            input_img = Image.open(
                f"{dataset_folder}/{folder}/{filename}").convert('L')
            input_img = np.array(input_img)

            dataset_inputs[f"{folder}-{part_id}"] = input_img

imgs = []
labels = []
for key, input_image in dataset_inputs.items():

    if key in dataset_labels:
        label_image = dataset_labels[key]

        # Crop center 500px
        bbox_image, (ymin, ymax, xmin, xmax) = crop_bbox(label_image)

        y_center = round((ymax + ymin) / 2)
        x_center = round((xmax + xmin) / 2)

        input_image = input_image[y_center - 250:y_center + 250, :]

        input_image = cv2.resize(
            input_image, (target_size, target_size)).astype(np.float32)

        assert input_image.shape == (target_size, target_size)

        imgs.append(input_image)
        # 0 as rare class in the paper
        labels.append(0)
    else:
        y_center = round(input_image.shape[0] / 2)

        input_image = input_image[y_center - 250:y_center + 250, :]

        input_image = cv2.resize(
            input_image, (target_size, target_size)).astype(np.float32)

        assert input_image.shape == (target_size, target_size)

        imgs.append(input_image)
        labels.append(1)

imgs = np.array(imgs)
labels = np.array(labels)


# Splitting
SEED = 42
x_train, x_val, y_train, y_val = train_test_split(imgs, labels,
                                                  random_state=SEED,
                                                  test_size=0.2,
                                                  stratify=labels)
print(x_train.shape, x_val.shape)

# %% --------------------------------------- Save as .npy ----------------
# Save
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val)
np.save("y_val.npy", y_val)
