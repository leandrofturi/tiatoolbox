import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tiatoolbox import logger
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import imread

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")

device = "cpu"

img_file_name = "/home/lfurlam/Documentos/PAD/E264810.czi"

# Tile prediction
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
)

output = bcc_segmentor.predict(
    [img_file_name],
    save_dir="sample_tile_results/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device=device,
    crash_on_exception=True,
)

logger.info(
    "Prediction method output is: %s, %s",
    output[0][0],
    output[0][1],
)
tile_prediction_raw = np.load(
    output[0][1] + ".raw.0.npy",
)  # Loading the first prediction [0] based on the output address [1]
logger.info(
    "Raw prediction dimensions: (%d, %d, %d)",
    tile_prediction_raw.shape[0],
    tile_prediction_raw.shape[1],
    tile_prediction_raw.shape[2],
)

# Simple processing of the raw prediction to generate semantic segmentation task
tile_prediction = np.argmax(
    tile_prediction_raw,
    axis=-1,
)  # select the class with highest probability
logger.info(
    "Processed prediction dimensions: (%d, %d)",
    tile_prediction.shape[0],
    tile_prediction.shape[1],
)

# showing the predicted semantic segmentation
tile = imread(img_file_name)
logger.info(
    "Input image dimensions: (%d, %d, %d)",
    tile.shape[0],
    tile.shape[1],
    tile.shape[2],
)

fig = plt.figure()
label_names_dict = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflamatory",
    3: "Necrosis",
    4: "Others",
}
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    (
        plt.imshow(tile_prediction_raw[:, :, i]),
        plt.xlabel(
            label_names_dict[i],
        ),
        ax.axes.xaxis.set_ticks([]),
        ax.axes.yaxis.set_ticks([]),
    )
fig.suptitle("Row prediction maps for different classes", y=0.65)

# showing processed results
fig2 = plt.figure()
ax1 = plt.subplot(1, 2, 1), plt.imshow(tile), plt.axis("off")
ax2 = plt.subplot(1, 2, 2), plt.imshow(tile_prediction), plt.axis("off")
fig2.suptitle("Processed prediction map", y=0.82)
plt.show()
