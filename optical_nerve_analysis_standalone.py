import sys
sys.path.append('/home/jovyan/projects/cellcount/')
import os
print(os.getcwd())
# os.chdir('/home/jovyan/projects/cellcount')
# print(os.getcwd())
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from skimage import io
from skimage import measure
from skimage import color
from PIL import Image
from pathlib import Path
from read_roi import read_roi_zip
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity, make_valid
from utils import *

from pickle import dump, load


species = 'Ft10'
path2kdrive = "/home/jovyan/kDrive/cellcount" # sur metea, il y a deux répertoires cellcount imbriqués
#path2kdrive = "/home/jovyan/kDrive" # tester sur lenovomic
path2data = os.path.join(path2kdrive, "cellcount/data/onunet/optical_nerve")
path2rawimages = os.path.join(path2kdrive, "cellcount/data/BVSA-Histology-Epoxy")
print("main path to data:", path2data)
path2test = os.path.join(path2data, "test")

archive_label = None
#archive_label = '03_10_2023'
if archive_label is None:
    path2prediction = os.path.join(path2data, "prediction", species)
else:
    path2prediction = os.path.join(path2data, "prediction_archive", archive_label, species)

print("using following prediction:", path2prediction)

# path to whole optical nerve image for 'species':
#path2dir = os.path.join("data/BVSA-Histology-Epoxy", species, "ON1_1")
if species == 'Aa06':
    imagefilename = "Aa06 - ON1_1 6_3 - 2 - 63x-Stitching-02.czi - Aa06 - ON1_1 6_3 - 2 - 63x-Stitching-02.czi #1.tif"
    foldername = os.path.join("ON1_1", "Aa06 - ON1_1 6_3")
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 1216.66/12285
elif species == 'Hr11':
    imagefilename = "Hr11 - ON1_1 6_4 - 2 - 63x overview-Stitching-04.tif"
    foldername = "ON1_1"
    px_ratio = 1032.15/10422
    # path2dir = os.path.join("tania/Bird_Visual_System_Atlas/BVSA-Histology-Epoxy", species, foldername) # old path
    path2dir = os.path.join(path2rawimages, species, foldername)
elif species == 'Mc12':
    imagefilename = "Mc12_ON2_7bis_2-63x-Stitching-03.czi - Mc12_ON2_7bis_2-63x-Stitching-03.czi #1.tif"
    foldername = "ON1_1"
    px_ratio = 1112.37/11232
    path2dir = os.path.join(path2rawimages, species, foldername)
    #imagefilename = "Mc12-ON1_w 6_3-1-63x_overview-Stitching-01.czi_#1.tif" ## previous sample
    #foldername = "Mc12-ON1_w 6_3"
    #path2dir = os.path.join("tania/Bird_Visual_System_Atlas/BVSA-Histology-Epoxy", species, foldername)
elif species == 'Cl': # data stored under a completely different path
    imagefilename = "ClCl PV 10W ON 3_3 - 1 - 63x overview-Stitching-01.czi #1.tif"
    foldername = "Cl PV ON"
    px_ratio = 1588.34/16038
    path2dir = os.path.join("tania/Epoxy_Histology/Pigeon", foldername)
elif species == 'Aa14': 
    imagefilename = "Aa14_ON2_4_4-1-63x-Stitching-05.czi - Aa14_ON2_4_4-1-63x-Stitching-05.czi #1.tif"
    foldername = "ON1_1"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 1228.76/18684
elif species == 'Hr21': 
    imagefilename = "Hr21_ON2_5-4_1-63x-Stitching-03.czi - Hr21_ON2_5-4_1-63x-Stitching-03.czi #1.tif"
    foldername = "ON1_1"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 1120.39/11313
elif species == 'Cj22': 
    imagefilename = "Cj22_ON1_5-1_1-63x-Stitching-05.czi - Cj22_ON1_5-1_1-63x-Stitching-05.czi #1.tif"
    foldername = "ON1_1"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 1671.23/16875
elif species == 'Du26': 
    imagefilename = "Du26_ON1_6bis_1-63x_2-Stitching-02.czi - Du26_ON1_6bis_1-63x_2-Stitching-02.czi #1.tif"
    foldername = "ON1_1"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 847.35/8556
elif species == 'Fp23': 
    imagefilename = "Fp23_ON2_7-3-63x-Stitching-01.czi - Fp23_ON2_7-3-63x-Stitching-01.czi #1.tif"
    foldername = "ON2"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 2510.86/25353
elif species == 'Ft10': 
    imagefilename = "Ft10_ON1_9-Stitching-01.czi - Ft10_ON1_9-Stitching-01.czi #1_crop.tif"
    foldername = "ON1_1"
    path2dir = os.path.join(path2rawimages, species, foldername)
    px_ratio = 1949.32/19683
else:
    raise ValueError('Unknown species or data not available for this species !')

# open file with whole optical nerve image:
path2image = os.path.join(path2kdrive, path2dir, imagefilename)
print(path2image)

Image.MAX_IMAGE_PIXELS = None # avoid the "DOS decompression bomb warning/error" by unlimiting the size of the displayed image
highlight_threshold = 10 # highlights in red axons with area greater than this value (micrometers^2)
#cellmarks_whole, df_axons = cellmarks_from_whole_mask(
cellmarks_whole = cellmarks_from_whole_mask(
    os.path.join(path2prediction, 'predicted_raw_mask.png'),
    path2image,
    contour_thr=0.7,
    discard_thr=0.09,
    highlight_thr=highlight_threshold, # Hr21: 12.944474, Aa14: 7.365529
    show_largest_only=True,
    #path2save_cellmarks=os.path.join(path2prediction, 'predicted_cellmarks_larger_than_15.png'),
    #path2save_cellmarks=os.path.join(path2prediction, 'predicted_cellmarks.png'),
    #path2save_biggest_only=os.path.join(path2prediction, f"largest_axons_on_mask_thr_{highlight_threshold}.png"),
    path2save_highlight_axons=os.path.join(path2prediction, f"largest_axons_on_mask_highlighted"),
    binary=True,
    binary_thr=0.7,
    #path2save_binary_mask=os.path.join(path2prediction, 'predicted_binary_mask.png'),
    px_ratio=px_ratio
)


# Saved data for further use:
# pickle_filename = os.path.join(path2prediction, f"cellmarks_{species}.pickle")
# with open(pickle_filename, 'bw') as f:
#     dump(cellmarks_whole, f)

