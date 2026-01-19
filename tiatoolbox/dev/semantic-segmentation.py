import shutil
from pathlib import Path
import numpy as np
import cv2
import tifffile
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.plot import plot_overlay


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


################################################################################

slide_path = "/home/lfurlam/Documentos/PAD/E264810.czi"

reader = WSIReader.open(slide_path)
tile = reader.read_bounds([74005,  4400,  2048,  2048])
arr = np.asarray(tile)
arr = np.squeeze(arr)
fname = f"tile_x{74005}_y{4400}_w{2048}_h{2048}.tif"
fname = f"/home/lfurlam/Documentos/PAD/E264810_{fname}"
tifffile.imwrite(
    str(fname),
    arr,
    compression="none",
    bigtiff=True if arr.nbytes > (2**32 - 1) else False,
)

shutil.rmtree("/home/lfurlam/Documentos/PAD/E264810_bound_result/")
output = bcc_segmentor.predict(
    [fname],
    save_dir="/home/lfurlam/Documentos/PAD/E264810_bound_result/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device=device,
    crash_on_exception=True,
)
up = np.load("{output}.raw.0.npy")
up = cv2.resize(up, (2048,  2048), interpolation=cv2.INTER_LINEAR)
plot_overlay(up, tile, label_names_dict, alpha=0.5)


################################################################################

shutil.rmtree("/home/lfurlam/Documentos/PAD/E264810_bound_result3/")
output = bcc_segmentor.predict(
    [tile],
    save_dir="/home/lfurlam/Documentos/PAD/E264810_bound_result3/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device=device,
    crash_on_exception=True,
    return_np=True
)
up3 = np.load("{output}.raw.0.npy")
up3 = cv2.resize(up3, (2048,  2048), interpolation=cv2.INTER_LINEAR)
plot_overlay(up3, tile, alpha=0.5)
