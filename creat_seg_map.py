import json
import os.path as p
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm.notebook import tqdm
import os
import pycocotools


def decode_segmap(label_mask, plot=False):


    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

label_colours = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


Map= [0, 15, 15, 2, 7, 5, 6, 11, 8, 9, 10, 0]
### Examine COCO format to see how it is done





## Conversion starts here


# set = 'train'
set = 'val'
VISDRONE_ANNOT_DIR = p.join('Visdrone/VisDrone2019-DET-train/annotations_origins')
VISDRONE_IM_DIR = p.join('Visdrone/VisDrone2019-DET-train/images')

# %%

OUTPUT_annot_black = p.join('Visdrone/VisDrone2019-DET-train/annotations_created')
OUTPUT_annot_color = p.join('Visdrone/VisDrone2019-DET-train/annotations_map')

# %%


#### Verify splitting func and CSV reading


# bla = '9999999_00877_d_0000402'.split('_')
# int(bla[0] + bla[1] + bla[3])

#
# csv = pd.read_csv(os.path.join(VISDRONE_ANNOT_DIR, '0000277_04401_d_0000560.txt'),
#                   names=['x', 'y', 'w', 'h', 'score', 'category', 'truncation', 'occlusion'])
#
# im = Image.open(p.join(VISDRONE_IM_DIR, '0000277_04401_d_0000560.jpg'))
# im_size = im.size
# im.close()
# img = np.zeros((im_size[1], im_size[0]))
# print(im_size[1], im_size[0])
# for r in csv.itertuples():
#     print(r.y, r.x, r.h, r.w, r.y+r.h, r.x+r.w, r.category)
#     img[r.y : r.y+r.h, r.x : r.x+r.w]=Map[r.category]
#
# cv2.imwrite(os.path.join(OUTPUT_annot_black, '0000277_04401_d_0000560.jpg'), img)
#
# i = cv2.imread(os.path.join(OUTPUT_annot_black, '0000277_04401_d_0000560.jpg'), cv2.IMREAD_GRAYSCALE)
#
# ou = decode_segmap(i, plot=False)
# cv2.imwrite(os.path.join(OUTPUT_annot_color, '0000277_04401_d_0000560.jpg'), ou)



## Convert!



# Iterate all existing CSV annotations
for root, dirs, files in os.walk(VISDRONE_ANNOT_DIR):
    for f in tqdm(files):
        (r, e) = p.splitext(f)
        if e != '.txt':
            continue

        # Read VisDrone source annotation
        print(f)
        csv = pd.read_csv(p.join(root, f), names=['x', 'y', 'w', 'h', 'score', 'category', 'truncation', 'occlusion'])


        # Get image size
        im = Image.open(p.join(VISDRONE_IM_DIR, r + '.jpg'))
        im_size = im.size
        im.close()

        img = np.zeros((im_size[1], im_size[0]))
        for k in csv.itertuples():
            img[k.y: k.y + k.h, k.x: k.x + k.w] = Map[k.category]

        cv2.imwrite(os.path.join(OUTPUT_annot_black, r + '.jpg'), img)

        i = cv2.imread(os.path.join(OUTPUT_annot_black, r + '.jpg'), cv2.IMREAD_GRAYSCALE)

        ou = decode_segmap(i, plot=False)
        cv2.imwrite(os.path.join(OUTPUT_annot_color, r + '.jpg'), ou)
