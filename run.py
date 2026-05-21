import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde


### all the lens side is averaged to the left side.
### info example ''bootstrap_pp_1_elderly_sphere'' 
# # refers to the {statistic process method}_{scenarios}_{group}_{lens}
# 'bootstrap_pp_1_elderly_sphere' is default value, as bootstrap
#  'scenarios' = pp , refers to smart phone, and laptop
#  'scenarios' = 'fppb', refers to free walk, and reading of smart phone, laptop, and distance projected screen
#  'lens' is default value, as sphere


plt.rcParams.update({
    "font.size": 14,       # 全局字体大小
    "axes.titlesize": 14,  # 子图标题
    "axes.labelsize": 14,  # 坐标轴标签
    "xtick.labelsize": 14, # x 轴刻度
    "ytick.labelsize": 14, # y 轴刻度
    "legend.fontsize": 14  # 图例
})



DEFAULT_DATA_PATH = Path(__file__).parent



def load_data(data_path=DEFAULT_DATA_PATH, data_name = None):
    data_path = f'{data_path}/{data_name}.pkl'
    with open(data_path, "rb") as f:
        reps = pickle.load(f)
        return reps


def _as_points(data, key):
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{key} should have shape (N, 2), got {arr.shape}")
    return arr


def _as_duration(data):
    duration = np.asarray(data["duration_XY"], dtype=float).reshape(-1)
    if np.any(duration < 0):
        raise ValueError("duration_XY should not contain negative values")
    return duration


def _filtered_points(points_XY, duration_XY, lens_edge_XY):
    mask = np.isfinite(points_XY).all(axis=1) & np.isfinite(duration_XY)
    points_XY = points_XY[mask]
    duration_XY = duration_XY[mask]

    if lens_edge_XY is not None:
        lens_edge_XY = np.asarray(lens_edge_XY, dtype=float)
        mask = (
            (points_XY[:, 0] >= lens_edge_XY[:, 0].min())
            & (points_XY[:, 0] <= lens_edge_XY[:, 0].max())
            & (points_XY[:, 1] >= lens_edge_XY[:, 1].min())
            & (points_XY[:, 1] <= lens_edge_XY[:, 1].max())
        )
        points_XY = points_XY[mask]
        duration_XY = duration_XY[mask]

    return points_XY, duration_XY


def _point_density(points_XY, duration_XY):
    if points_XY.shape[0] < 3:
        return np.ones(points_XY.shape[0], dtype=float)

    weights = np.asarray(duration_XY, dtype=float).reshape(-1)
    if np.isclose(weights.sum(), 0):
        weights = None

    try:
        xy = np.vstack([points_XY[:, 0], points_XY[:, 1]])
        return gaussian_kde(xy, weights=weights)(xy)
    except Exception:
        return np.ones(points_XY.shape[0], dtype=float)


def _format_mm(x, pos):
    return f"{x * 1000:.0f}"


def plot_representor_points(
    data,
    output_path=None,
    density_vmin=0,
    density_vmax=60000,
    point_size=2,
):
    points_XY = _as_points(data, "points_XY")
    duration_XY = _as_duration(data)
    lens_edge_XY = _as_points(data, "lens_edge_XY")
    center = np.asarray(data.get("center", [[0, 0]]), dtype=float).reshape(-1, 2)
    side = data.get("side", "")
    info = data.get("info", "representor_x0")

    points_XY, duration_XY = _filtered_points(points_XY, duration_XY, lens_edge_XY)
    density = _point_density(points_XY, duration_XY)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    title = f"{info}"

    cmap_name = "viridis"
    norm_density = mpl.colors.Normalize(vmin=density_vmin, vmax=density_vmax)

    ax.scatter(
        points_XY[:, 0],
        points_XY[:, 1],
        c=density,
        s=point_size,
        cmap=cmap_name,
        norm=norm_density,
        alpha=1,
        label="lens utility",
    )
    ax.plot(
        lens_edge_XY[:, 0],
        lens_edge_XY[:, 1],
        linestyle="--",
        linewidth=0.8,
        color="black",
        label="lens edge",
    )
    ax.scatter(
        center[:, 0],
        center[:, 1],
        c="red",
        marker="+",
        zorder=3,
        s=100,
        label="pupil center",
    )
    ax.set_title(f"{title}\n{side}", fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(_format_mm))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_mm))

    mappable_density = mpl.cm.ScalarMappable(
        norm=norm_density,
        cmap=plt.get_cmap(cmap_name),
    )
    cbar_density = fig.colorbar(
        mappable_density,
        ax=ax,
        orientation="vertical",
        fraction=0.08,
        pad=0.04,
        aspect=5,
    )


    cbar_density.set_label("Density")
    cbar_density.set_ticks(np.arange(density_vmin, density_vmax + 1, 10000))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def main(data_name, density_vmax):
    data = load_data(data_name=data_name)
    out_pdf = DEFAULT_DATA_PATH / f"{data_name}.pdf"
    out_png = DEFAULT_DATA_PATH / f"{data_name}.png"
    fig, _ = plot_representor_points(data, output_path=out_pdf, density_vmax= density_vmax)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    data_name = 'bootstrap_fppb_0_young_sphere'
    data_name = 'bootstrap_pp_1_elderly_sphere'
    data_name = 'bootstrap_fppb_1_elderly_sphere'
    if data_name == 'bootstrap_pp_1_elderly_sphere':
        main(data_name, density_vmax = 60000)
    if data_name == 'bootstrap_fppb_1_elderly_sphere':
        main(data_name, density_vmax = 40000)
    if data_name == 'bootstrap_fppb_0_young_sphere':
        main(data_name, density_vmax = 40000)
