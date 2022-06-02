"""
Step 3. Analyze each cell of the grid to search for lysis plaque.
Clean the dataset
"""
import os
import pandas as pd
import shutil

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_nn/train_set2"
os.chdir(directory)

d = pd.read_csv("D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_nn/new_train_set.csv", sep=";")

print(f"Before cleaning : {len(os.listdir(directory))} images in directory.")

print(d["file"])

i = 0
for k, ifl in enumerate(sorted(os.listdir(directory))):
    
    if ifl in d["file"].to_numpy():
        continue
    else:
        os.remove(ifl)
        # i += 1
    
print(i)
print(f"After cleaning : {len(os.listdir(directory))} images in directory.")
