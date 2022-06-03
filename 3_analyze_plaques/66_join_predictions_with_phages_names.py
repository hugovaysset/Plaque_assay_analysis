import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

import napari

predictions_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions_bin"
phage_name_table = pd.read_csv("D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/phage_names.csv", sep=";")

target_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions_final"
joined, not_joined = 0, 0
for f in sorted(os.listdir(predictions_dir)):

    if f.endswith(".csv"):
        plate_index = f[-5]  # index of the plate : A, B, C, D or nothing
        predictions = pd.read_csv(predictions_dir + "/" + f, sep=";")

        if plate_index in ["A", "a", "B", "b", "C", "c", "D", "d"]:
            phage_names = phage_name_table[phage_name_table["plate"] == plate_index.upper()]["phage_name"].to_numpy()
            predictions["phi"] = phage_names
            predictions.to_csv(target_dir + "/joined/" + f, sep=";")
            joined += 1
        else:
            predictions.to_csv(target_dir + "/not_joined/" + f, sep=";")
            not_joined += 1

print(f"Joined {joined}/{joined + not_joined} files.")
