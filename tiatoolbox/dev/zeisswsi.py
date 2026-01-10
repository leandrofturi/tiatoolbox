import numpy as np
from tiatoolbox.wsicore.wsireader import WSIReader
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")

slide_path = "/home/lfurlam/Documentos/PAD/E264810.czi"

slide = WSIReader.open(slide_path)

print("")
print("Níveis disponíveis:", slide.info.level_count)
print("Dimensão em cada nível:", slide.info.level_dimensions)
print("Downsampling por nível:", slide.info.level_downsamples)

region = slide.read_region(
    location=(74005, 4400),
    level=0,
    size=(2056, 2464)
)

region = Image.fromarray(region, mode="RGB")

plt.imshow(region)
plt.axis("off")
plt.show()

thumb_np = slide.slide_thumbnail(resolution=8.0, units="power")
thumb_np.shape
thumb = Image.fromarray(thumb_np, mode="RGB")
thumb.size

plt.imshow(thumb)
plt.axis("off")
plt.show()


mask = slide.tissue_mask(resolution=8.0, units="power")
thumb_np = mask.slide_thumbnail(resolution=8.0, units="power").astype(np.uint8)
coords = np.argwhere(thumb_np > 0)
first_y, first_x = coords[0]   # (linha, coluna)
print(first_x, first_y)
