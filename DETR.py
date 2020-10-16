from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torchvision.datasets as D
import numpy as np
torch.set_grad_enabled(False)
from joblib import Parallel, delayed

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

import glob
from tqdm.notebook import tqdm

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
#     T.Resize(1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()
model = model.cuda()

# meta

batch_size = 4
INPUT_FOLDER = 'Visdrone/val/image/'
dataset = D.ImageFolder(INPUT_FOLDER, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
           pin_memory=True, drop_last=False)

OUT_DIR = 'Visdrone/val/result_DETR/'
os.makedirs(OUT_DIR, exist_ok=True)
# Go over all images, run model, get results
index = 0
for data in tqdm(dataloader):

    # mean-std normalize the input image (batch-size: 1)
    img = data[0]
    out = model(img.cuda())

    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    results = postprocessor(out, [torch.as_tensor(img.shape[-2:]) for i in range(batch_size)])

    for res in results:
        # We extract the segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(res["segments_info"])
        # Panoptic predictions are stored in a special format png
        panoptic_seg = Image.open(io.BytesIO(res['png_string']))
        final_w, final_h = panoptic_seg.size
        # We convert the png into an segment id map
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

        # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i][
                "isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

        # Finally we visualize the prediction
        v = Visualizer(np.array(Image.open(dataset.imgs[index][0]).resize((final_w, final_h)))[:, :, ::-1], meta,
                       scale=1.0)
        v._default_font_size = 20
        v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
        visualization = Image.fromarray(v.get_image())
        visualization.save(os.path.join(OUT_DIR, 'vis_%04d.jpg' % index))
        index += 1

# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(out["pred_masks"][keep]):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask, cmap="cividis")
    ax.axis('off')
fig.tight_layout()

# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

from copy import deepcopy

# We extract the segments info and the panoptic result from DETR's prediction
segments_info = deepcopy(result["segments_info"])
# Panoptic predictions are stored in a special format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
final_w, final_h = panoptic_seg.size
# We convert the png into an segment id map
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

# Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
for i in range(len(segments_info)):
    c = segments_info[i]["category_id"]
    segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else \
    meta.stuff_dataset_id_to_contiguous_id[c]

# # Finally we visualize the prediction
# v = Visualizer(np.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
# v._default_font_size = 20
# v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
# Image.fromarray(v.get_image())
