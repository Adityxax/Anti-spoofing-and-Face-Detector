import os
import random
import shutil
from itertools import islice

outputFolderPath = "Datasets/SplitData"
inputFolderPath = "Datasets/DataCollect"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Clear and recreate output directory
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath, exist_ok=True)

# -------Directories to create-----
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

# -------Get the Names------
listNames = os.listdir(inputFolderPath)
print(listNames)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# -------Shuffle-----
random.shuffle(uniqueNames)
print(uniqueNames)

# -------Find the number of images for each folder-----
lenData = len(uniqueNames)
print(f'Total Images: {lenData}')
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])
print(f'Total Images:{lenData}\n Split: {lenTrain},{lenVal},{lenTest}')

# ----Put the remaining images in Training-----
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain += remaining

# -------Split the list-----
lengthToSplit = (lenTrain, lenVal, lenTest)
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData}\n Split: {len(Output[0])},{len(Output[1])},{len(Output[2])}')

# -------Copy the file-----
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        img_path = f"{inputFolderPath}/{fileName}.jpg"
        label_path = f"{inputFolderPath}/{fileName}.txt"

        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.copy(img_path, f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
            shutil.copy(label_path, f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")
        else:
            print(f"⚠ Skipping {fileName}: missing .jpg or .txt file")

print("Split Process Completed......")

# --------Creating data.yaml file------
# Paths in data.yaml are relative to the 'path' key
dataYaml = (f"path: {os.path.abspath(outputFolderPath)}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n\n"
            f"nc: {len(classes)}\n"
            f"names: {classes}")

with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print(f"data.yaml file created at {outputFolderPath}/data.yaml")

print("Data.yaml file created...")