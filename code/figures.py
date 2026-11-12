import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
import items

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
VENDOR = {"anthropic": "#2a78d6", "gemini": "#eb6834", "openai": "#1baf7a", "qwen": "#6250d6"}
CANVAS = {"1024x768": "#86b6ef", "2048x1536": "#3987e5", "4096x3072": "#1c5cab", "8000x6000": "#0d366b"}
LOST = (("rejected", "image rejected", "#52514e"), ("cannot_tell", "said it cannot tell", "#c3c2b7"),
        ("other_wrong", "wrong, confidence below 70", "#86b6ef"),
        ("confident_wrong", "wrong, confidence 70 or more", "#e34948"))
STRATEGY = {"as_is": ("o", "as is"), "max": ("^", "highest setting"), "tile2": ("s", "2×2 tiles"),
            "tile3": ("p", "3×3 tiles"), "zoom": ("D", "two-step zoom"), "check": ("*", "zoom check"),
            "oracle": ("X", "oracle")}
NAMES = {"claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet", "gemini-3.5-flash-lite": "Flash-Lite",
         "gemini-3.8-flash": "Flash", "gpt-6-luna": "Luna", "gpt-6.1-sol": "Sol", "qwen3.8-27b": "Qwen"}
CURVE_SETTING = {"anthropic": "native", "gemini": "high", "openai": "high", "qwen": "default"}
SETTING_NAMES = {"native": "native", "old-tier": "standard", "low": "low", "medium": "medium", "high": "high",
                 "ultra_high": "ultra high", "auto": "auto", "default": "default"}
TIERS = (("small", ("claude-haiku-5-5", "gemini-3.5-flash-lite", "gpt-6-luna")),
         ("mid", ("claude-sonnet-5-5", "gemini-3.8-flash", "gpt-6.1-sol")))


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def power_label(value, _):
    return f"{value:g}" if value >= 1 else f"1/{round(1 / value)}"


def log2_axis(axis, ticks=None):
    axis.set_xscale("log", base=2)
    if ticks:
        axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(power_label))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def example(values):
    rows = {row["id"]: row for row in config.read_jsonl(config.ITEMS / "eye.jsonl")}
    figure = plt.figure(figsize=(3.4, 1.55))
    grid = figure.add_gridspec(2, 4, width_ratios=[2.2, 1, 1, 1], wspace=0.08, hspace=0.25)
    whole = figure.add_subplot(grid[:, 0])
    item = rows["eye-2048x1536-c040-00"]
    whole.imshow(Image.open(config.IMAGES / "eye" / f"{item['id']}.jpg"), interpolation="none")
    whole.set_title("2048 × 1536 canvas", fontsize=6)
    for index, cap in enumerate((4, 6, 9, 13, 19, 40)):
        item = rows[f"eye-2048x1536-c{cap:03d}-00"]
        x0, y0, x1, y1 = item["label"]
        middle_x, middle_y, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 3 * (y1 - y0)) * 0.62
        crop = Image.open(config.IMAGES / "eye" / f"{item['id']}.jpg").crop(
            (round(middle_x - half), round(middle_y - half / 2), round(middle_x + half), round(middle_y + half / 2)))
        axis = figure.add_subplot(grid[index // 3, 1 + index % 3])
        axis.imshow(crop, interpolation="none")
        axis.set_title(f"{cap} px", fontsize=6, pad=2)
    for axis in figure.axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    return figure


def eye_curves(values):
    models = list(items.MODELS)
    figure, axes = plt.subplots(2, len(models), figsize=(7.0, 3.0), sharey=True)
    cells = values["eye"]["cells"]
    for column, model in enumerate(models):
        setting = CURVE_SETTING[items.MODELS[model]["vendor"]]
        fit = values["eye"]["fits"][f"{model}|{setting}"]
        for canvas, color in CANVAS.items():
            rows = [row for row in cells if row["model"] == model and row["setting"] == setting
                    and row["canvas"] == canvas and row["exact"] is not None]
            if not rows:
                continue
            exact = [row["exact"] for row in rows]
            axes[0, column].plot([row["cap"] for row in rows], exact, color=color, marker="o", markersize=2.5,
                                 label=canvas.replace("x", " × "))
            axes[1, column].plot([row["tokens"] for row in rows], exact, color=color, marker="o", markersize=2.5)
        beta = fit["received"]["beta"]
        if beta:
            grid = np.linspace(np.log2(min(r["tokens"] for r in cells if r["model"] == model and r["tokens"])),
                               np.log2(max(r["tokens"] for r in cells if r["model"] == model and r["tokens"])), 100)
            axes[1, column].plot(2 ** grid, 1 / (1 + np.exp(-(beta[0] + beta[1] * grid))), color=INK,
                                 linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=3)
        axes[0, column].set_title(f"{NAMES[model]}\n{SETTING_NAMES[setting]}")
        log2_axis(axes[0, column], [4, 16, 64])
        log2_axis(axes[1, column], [1 / 32, 1 / 4, 2])
        for row in (0, 1):
            axes[row, column].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("exact reads")
    axes[1, 0].set_ylabel("exact reads")
    middle = len(models) // 2
    axes[0, middle].set_xlabel("cap height in original pixels")
    axes[1, middle].set_xlabel("cap height in image tokens received")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, [f"{label} canvas" for label in labels], loc="lower center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(h_pad=0.8, w_pad=0.3, rect=(0, 0.05, 1, 1))
    return figure


def eight_k(values):
    rows = values["eye"]["eight_k"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(3.4, 2.9), sharey=True)
    labels = [f"{NAMES[row['model']]}, {SETTING_NAMES[row['setting']]}" for row in rows]
    places = np.arange(len(rows))[::-1]
    for place, row in zip(places, rows):
        color = VENDOR[items.MODELS[row["model"]]["vendor"]]
        if row["rejected"]:
            for axis in (left, right):
                axis.text(0.02, place, "rejected", transform=axis.get_yaxis_transform(), va="center", fontsize=6,
                          color=MUTED)
            continue
        left.barh(place, row["readable_cap"], height=0.62, color=color)
        left.text(row["readable_cap"], place, f" {row['readable_cap']:.0f}", va="center", fontsize=5.5, color=INK)
        right.barh(place, row["tokens"], height=0.62, color=color)
        right.text(row["tokens"], place, f" {row['tokens']:,}", va="center", fontsize=5.5, color=INK)
    left.set_yticks(places, labels)
    left.set_xlabel("smallest readable\ncap height (px)")
    right.set_xlabel("image tokens\nfor the photo")
    right.set_xscale("log")
    right.set_xlim(150, 60000)
    left.set_xlim(0, 30)
    for axis in (left, right):
        axis.grid(axis="y", visible=False)
    figure.tight_layout(w_pad=0.4)
    return figure


def lost(values):
    figure, axis = plt.subplots(figsize=(3.4, 1.9))
    models = [model for _, group in TIERS for model in group]
    places = np.arange(len(models))[::-1]
    for place, model in zip(places, models):
        entry = values["photos"]["lost"][model]
        start = 0.0
        for key, label, color in LOST:
            share = entry[key] / entry["lost"] if entry["lost"] else 0
            axis.barh(place, share, left=start, height=0.62, color=color, edgecolor="#fcfcfb", linewidth=1,
                      label=label if place == places[0] else None)
            start += share
        axis.text(1.01, place, f"{entry['lost']}", va="center", fontsize=5.5, color=MUTED,
                  transform=axis.get_yaxis_transform())
    axis.text(1.01, places[0] + 0.75, "n", va="center", fontsize=5.5, color=MUTED,
              transform=axis.get_yaxis_transform())
    axis.set_yticks(places, [NAMES[model] for model in models])
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.set_xlabel("share of questions")
    axis.grid(axis="y", visible=False)
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.42, 1.0), handlelength=1.2,
                columnspacing=0.8)
    figure.tight_layout()
    return figure


def accuracy_tokens(values):
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    table = values["photos"]["strategies"]
    for axis, (tier, models) in zip(axes, TIERS):
        for model in models:
            color = VENDOR[items.MODELS[model]["vendor"]]
            entries = table[model]
            order = sorted((name for name in entries if name not in ("check", "oracle")),
                           key=lambda name: entries[name]["tokens"])
            axis.plot([entries[name]["tokens"] for name in order], [entries[name]["accuracy"] for name in order],
                      color=color, linewidth=0.9, alpha=0.7, label=NAMES[model])
            for name, entry in entries.items():
                marker, _ = STRATEGY[name]
                size = 9 if name == "check" else 4.5
                hollow = name == "check"
                axis.errorbar(entry["tokens"], entry["accuracy"],
                              yerr=[[entry["accuracy"] - entry["ci"][0]], [entry["ci"][1] - entry["accuracy"]]],
                              color=color, marker=marker, markersize=size, linewidth=0.6, capsize=0,
                              markerfacecolor="none" if hollow else color,
                              markeredgecolor=color if hollow else "#fcfcfb", markeredgewidth=0.8 if hollow else 0.6,
                              zorder=3 if hollow else 2)
        axis.set_xscale("log")
        axis.set_xlabel("image tokens per question (mean)")
        axis.set_title(f"{tier} models")
        axis.legend(frameon=False, loc="lower right", handlelength=1.6)
    axes[0].set_ylabel("accuracy")
    shapes = [plt.Line2D([], [], color=MUTED, marker=marker, linestyle="", markersize=7 if name == "check" else 4.5,
                         markerfacecolor="none" if name == "check" else MUTED, label=label)
              for name, (marker, label) in STRATEGY.items()]
    figure.legend(handles=shapes, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def main():
    style()
    values = json.loads((config.RESULTS / "values.json").read_text())
    folder = config.FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    for name, draw in (("example", example), ("eye_curves", eye_curves), ("eight_k", eight_k), ("lost", lost),
                       ("accuracy_tokens", accuracy_tokens)):
        figure = draw(values)
        figure.savefig(folder / f"{name}.pdf")
        plt.close(figure)
        print(folder / f"{name}.pdf")


if __name__ == "__main__":
    main()
