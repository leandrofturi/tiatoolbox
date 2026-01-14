import shutil
import logging
from pathlib import Path

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor
)

device = "cpu"

tiles_dir = Path("/home/lfurlam/Documentos/PAD/E264810_tiles")

# Tile prediction
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
)

tile_files = sorted(tiles_dir.glob("tile_*_x*_y*_w*_h*.tif"))

shutil.rmtree("/home/lfurlam/Documentos/PAD/E264810_tiles_results/")
output = bcc_segmentor.predict(
    tile_files,
    save_dir="/home/lfurlam/Documentos/PAD/E264810_tiles_results/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    # patch_input_shape=[2464, 2048],   # [h=2056, w=2464]
    # patch_output_shape=[1232, 1024],  # metade do input
    # stride_shape=[608, 512],  # 50% overlap no output
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device=device,
    crash_on_exception=True,
)

with open("/home/lfurlam/Documentos/PAD/E264810_tiles_results/results.txt", 'w') as file:
    file.write('\n'.join(output) + '\n')
