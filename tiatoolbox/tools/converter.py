import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from aicspylibczi import CziFile
import tifffile
from matplotlib.colors import ListedColormap


def convert_tiles(
    czi_file: str, output_path: str | Path = None,
    C: int | None = None, S: int | None = None,
    use_tqdm: bool = True
):
    czi_path = Path(czi_file)
    if output_path is None:
        output_path = czi_path.parent / f"{czi_path.stem}_tiles"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    czi = CziFile(str(czi_path))
    if S:
        bboxes = czi.get_all_mosaic_tile_bounding_boxes(S=S)
    else:
        bboxes = czi.get_all_mosaic_tile_bounding_boxes()

    dims = czi.get_dims_shape()
    channel = C or min([dim['C'][0] for dim in dims])

    iterator = bboxes.items()
    if use_tqdm:
        iterator = tqdm(
            bboxes.items(), total=len(bboxes), desc="Exportando tiles",
            unit="tile"
        )

    for i, (_, box) in enumerate(iterator, start=1):
        read_kwargs = {
            "region": (box.x, box.y, box.w, box.h),
            "scale_factor": 1.0,
            "C": channel
        }

        tile = czi.read_mosaic(**read_kwargs)
        arr = np.asarray(tile)
        arr = np.squeeze(arr)
        fname = f"tile_{i:05d}_x{box.x}_y{box.y}_w{box.w}_h{box.h}.tif"
        fname = output_path / fname
        tifffile.imwrite(
            str(fname),
            arr,
            compression="none",
            bigtiff=True if arr.nbytes > (2**32 - 1) else False,
        )


def save_mosaic_overlay_tiff(
    canvas_rgb, canvas_class, out_path, alpha=0.5, compression='lzma'
):
    colors_hex = [
        "#1f77b4",  # Tumour
        "#ff7f0e",  # Stroma
        "#2ca02c",  # Inflammatory
        "#d62728",  # Necrosis
        "#9467bd",  # Others
    ]
    cmap = ListedColormap(colors_hex)

    if np.issubdtype(canvas_rgb.dtype, np.integer):
        base_float = (
            canvas_rgb.astype(np.float32) / np.iinfo(canvas_rgb.dtype).max
        )
    else:
        base_float = canvas_rgb.astype(np.float32)
        base_float = np.clip(base_float, 0.0, 1.0)

    lut = np.array([cmap(i)[:3] for i in range(cmap.N)], dtype=np.float32)
    overlay_rgb = lut[canvas_class]

    result = alpha * overlay_rgb + (1.0 - alpha) * base_float
    result = np.clip(result, 0.0, 1.0)

    result_u8 = (result * 255).astype(np.uint8)
    tifffile.imwrite(
        str(out_path),
        result_u8,
        compression=compression,
        photometric='rgb'
    )


def stitch_from_tiles(
    czi_path: str | Path,
    tiles_dir: str | Path,
    output_path: str | Path | None = None,
    output_class_path: str | Path | None = None,
    scale_factor: float = 0.1,
    compression: str | None = "lzma",  # 'none', 'lzw', 'zlib', 'lzma'
    use_tqdm: bool = True
):
    czi_path = Path(czi_path)
    tiles_dir = Path(tiles_dir)

    if output_path is None:
        output_path = f"{czi_path.stem}_stitch_{int(scale_factor*100)}pct.tif"
        output_path = czi_path.parent / output_path
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_re = re.compile(
        r".*?_x(-?\d+)_y(-?\d+)_w(\d+)_h(\d+)\.tif$", re.IGNORECASE
    )

    czi = CziFile(str(czi_path))
    bb = czi.get_mosaic_bounding_box()
    W_full, H_full = bb.w, bb.h
    X_full, Y_full = bb.x, bb.y

    W_target = max(1, int(W_full * scale_factor))
    H_target = max(1, int(H_full * scale_factor))

    tile_files = sorted(tiles_dir.glob("tile_*_x*_y*_w*_h*.tif"))
    if not tile_files:
        raise FileNotFoundError(f"Nenhum tile encontrado em {tiles_dir}")

    first = tifffile.imread(str(tile_files[0]))
    dtype = first.dtype
    channels = first.shape[-1]
    canvas = np.zeros((H_target, W_target, channels), dtype=dtype)
    canvas_class = None
    if output_class_path is not None:
        canvas_class = np.zeros((H_target, W_target), dtype=dtype)

    def resize_nearest(
        img: np.ndarray | None, out_h: int, out_w: int
    ) -> np.ndarray:
        if img is None:
            return None
        in_h, in_w = img.shape[:2]
        if in_h == out_h and in_w == out_w:
            return img
        yy = (np.linspace(0, in_h - 1, out_h)).astype(np.int64)
        xx = (np.linspace(0, in_w - 1, out_w)).astype(np.int64)
        if img.ndim == 2:
            return img[np.ix_(yy, xx)]
        return img[np.ix_(yy, xx, np.arange(img.shape[2]))]

    iterator = tile_files
    if use_tqdm:
        iterator = tqdm(tile_files, desc="Recompondo mosaico", unit="tile")

    skipped = 0
    placed = 0

    for idx, f in enumerate(iterator):
        m = tile_re.match(f.name)
        if not m:
            skipped += 1
            continue

        x, y, w, h = map(int, m.groups())
        x, y = x - X_full, y - Y_full

        tile = tifffile.imread(str(f))
        if tile.ndim == 3 and channels != tile.shape[-1]:
            skipped += 1
            continue

        tile_class = None
        if output_class_path is not None:
            class_path = Path(f"{output_class_path}/{idx}.raw.0.npy")
            if class_path.is_file():
                tile_prediction_raw = np.load(class_path)
                pred_classes = np.argmax(tile_prediction_raw, axis=-1)
                if pred_classes.shape[:2] == tile.shape[:2]:
                    tile_class = pred_classes

        w_t = max(1, int(w * scale_factor))
        h_t = max(1, int(h * scale_factor))
        tile_s = resize_nearest(tile, h_t, w_t)
        tile_class_s = resize_nearest(tile_class, h_t, w_t)

        x_t = int(x * scale_factor)
        y_t = int(y * scale_factor)

        x_end = min(x_t + w_t, W_target)
        y_end = min(y_t + h_t, H_target)
        w_eff = x_end - x_t
        h_eff = y_end - y_t
        if w_eff <= 0 or h_eff <= 0:
            skipped += 1
            continue

        canvas[y_t:y_end, x_t:x_end, :] = tile_s[:h_eff, :w_eff, :]
        if canvas_class is not None and tile_class_s is not None:
            canvas_class[y_t:y_end, x_t:x_end] = tile_class_s[:h_eff, :w_eff]

        placed += 1

    tifffile.imwrite(
        str(out_path),
        canvas,
        compression=compression,
        bigtiff=True if canvas.nbytes > (2**32 - 1) else False
    )

    if canvas_class is not None:
        output_class_path = (
            f"{czi_path.stem}_stitch_{int(scale_factor*100)}pct_class.tif"
        )
        output_class_path = czi_path.parent / output_class_path
        save_mosaic_overlay_tiff(
            canvas, canvas_class, output_class_path,
            alpha=0.5, compression=compression
        )


convert_tiles("/home/lfurlam/Documentos/PAD/E264810.czi", S=0)

stitch_from_tiles(
    czi_path="/home/lfurlam/Documentos/PAD/E264810.czi",
    tiles_dir="/home/lfurlam/Documentos/PAD/E264810_tiles",
    output_class_path="/home/lfurlam/Documentos/PAD/E264810_tiles_results",
    scale_factor=0.05,
    compression="lzma",
    use_tqdm=True
)
