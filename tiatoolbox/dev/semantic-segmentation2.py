import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from tiatoolbox.utils.misc import imread

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["figure.facecolor"] = "white"


tile_prediction_raw = np.load(
    "/home/lfurlam/Documentos/PAD/E264810_tiles_results/0.raw.0.npy"
)
print(
    "Raw prediction dimensions:",
    tile_prediction_raw.shape[0],
    tile_prediction_raw.shape[1],
    tile_prediction_raw.shape[2],
)

tile_prediction = np.argmax(
    tile_prediction_raw,
    axis=-1,
)
print(
    "Processed prediction dimensions:",
    tile_prediction.shape[0],
    tile_prediction.shape[1],
)

tile_file = "/home/lfurlam/Documentos/PAD/E264810_tiles/tile_00001_x-236851_y19958_w2056_h2464.tif"
tile = imread(tile_file)
print(
    "Input image dimensions:",
    tile.shape[0],
    tile.shape[1],
    tile.shape[2],
)


label_names_dict = {0:"Tumour", 1:"Stroma", 2:"Inflamatory", 3:"Necrosis", 4:"Others"}

def plot_probability_maps(tile_prediction_raw, label_names_dict):
    H, W, C = tile_prediction_raw.shape
    assert C == 5, f"Esperava 5 canais, recebi {C}."
    fig, axes = plt.subplots(1, C, figsize=(4*C, 4), constrained_layout=True)
    # Se axes vier como um único objeto (quando C==1), force array
    if C == 1:
        axes = np.array([axes])
    for i in range(C):
        ax = axes[i]
        im = ax.imshow(tile_prediction_raw[:, :, i], cmap="magma")
        ax.set_title(label_names_dict.get(i, f"Class {i}"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Barra de cores por subplot
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Probabilidade", fontsize=9)
    plt.suptitle("Mapas de probabilidade por classe", y=1.02, fontsize=12)
    plt.show()


# 2) Plota a classe vencedora (argmax) com colormap discreto e legenda
def plot_argmax_mask(tile_prediction_raw, tile, label_names_dict):
    pred_classes = np.argmax(tile_prediction_raw, axis=-1)  # (H, W)
    # Paleta discreta (5 cores distintas)
    colors = [
        "#1f77b4",  # Tumour
        "#ff7f0e",  # Stroma
        "#2ca02c",  # Inflamatory
        "#d62728",  # Necrosis
        "#9467bd",  # Others
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(0, 6)  # 0..5
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    # Imagem original
    axes[0].imshow(tile)
    axes[0].set_title("Tile original")
    axes[0].axis("off")
    # Máscara discreta
    im = axes[1].imshow(pred_classes, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Classe vencedora (argmax)")
    axes[1].axis("off")
    # Construção da legenda discreta
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="k", label=label_names_dict.get(i, f"Class {i}"))
        for i in range(len(colors))
    ]
    axes[1].legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
                   ncol=3, fontsize=9, frameon=True)
    plt.show()


# 3) (Opcional) Overlay da máscara na imagem original com alpha
def plot_overlay(tile_prediction_raw, tile, label_names_dict, alpha=0.5):
    pred_classes = np.argmax(tile_prediction_raw, axis=-1)
    # Mesmo colormap discreto
    colors = [
        "#1f77b4",  # Tumour
        "#ff7f0e",  # Stroma
        "#2ca02c",  # Inflamatory
        "#d62728",  # Necrosis
        "#9467bd",  # Others
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(0, 6)
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(tile)
    ax.imshow(pred_classes, cmap=cmap, norm=norm, alpha=alpha)
    ax.set_title("Overlay: classe vencedora sobre a tile")
    ax.axis("off")
    # Legenda
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="k", label=label_names_dict.get(i, f"Class {i}"))
        for i in range(len(colors))
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=9, frameon=True)
    plt.show()


plot_probability_maps(tile_prediction_raw, label_names_dict)
plot_argmax_mask(tile_prediction_raw, tile, label_names_dict)
plot_overlay(tile_prediction_raw, tile, label_names_dict, alpha=0.5)
