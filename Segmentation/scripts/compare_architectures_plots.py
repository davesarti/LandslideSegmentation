import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def get_first_key(d, keys, default=None):
	for k in keys:
		if k in d and d[k] is not None:
			return d[k]
	return default


def load_metrics(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	train_loss = get_first_key(data, ["train_loss", "loss_train", "train"], [])
	val_loss = get_first_key(data, ["val_loss", "loss_val", "val"], [])
	train_iou = get_first_key(data, ["train_iou", "train_IoU", "train_mIoU", "miou_train", "iou_train"], [])
	val_iou = get_first_key(data, ["val_iou", "val_IoU", "val_mIoU", "miou_val", "iou_val"], [])
	prefix = data.get("prefix", Path(json_path).stem.replace("_metrics", ""))
	return prefix, train_loss, val_loss, train_iou, val_iou


def style_axes(fig, axes):
	fig.set_facecolor("white")
	fig.patch.set_alpha(1)
	for ax in axes:
		ax.set_facecolor("white")
		ax.minorticks_off()
		ax.grid(True, which="major", linestyle="-", alpha=0.5)
		ax.grid(True, which="minor", linestyle=":", alpha=0.35)
		ax.tick_params(axis="both", which="both", length=0, labelsize=12)
		for spine in ax.spines.values():
			spine.set_visible(False)


def main():
	root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description="Confronta le architetture di segmentazione.")
	artifacts_dir = root / "artifacts"
	parser.add_argument("--metrics-dir", type=Path, default=artifacts_dir / "metrics", help="Directory dei JSON delle metriche.")
	parser.add_argument("--output-dir", type=Path, default=artifacts_dir / "plots", help="Directory in cui salvare i plot.")
	args = parser.parse_args()
	metrics_dir = args.metrics_dir
	out_dir = args.output_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	# Modelli da confrontare
	paths = [
		metrics_dir / "0.7unet_bce_dice_metrics.json",
		metrics_dir / "0.7attunet_bce_dice_metrics.json",
		metrics_dir / "0.3swin_bce_metrics.json"
	]
	for p in paths:
		if not p.exists():
			raise FileNotFoundError(f"File non trovato: {p}")

	# Caricamento serie
	series = []  # (label, e_tr, tr, e_vl, vl, e_vi, vi)
	for p in paths:
		label, tr, vl, _ti, vi = load_metrics(p)
		e_tr = list(range(1, len(tr) + 1))
		e_vl = list(range(1, len(vl) + 1))
		e_vi = list(range(1, len(vi) + 1)) if vi else []
		series.append((label, e_tr, tr, e_vl, vl, e_vi, vi))

	# Palette colori
	palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

	# Figura: Train/Val loss
	fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
	style_axes(fig, axes)

	# Train
	ax = axes[0]
	for i, (lbl, e_tr, tr, *_rest) in enumerate(series):
		ax.plot(e_tr, tr, label=f"{lbl} train", color=palette[i % len(palette)])
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	# Val
	ax = axes[1]
	for i, (lbl, *_rest) in enumerate(series):
		_, _, _, e_vl, vl, _, _ = series[i]
		ax.plot(e_vl, vl, label=f"{lbl} val", color=palette[i % len(palette)])
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_loss = out_dir / "compare_train_val_loss_attention_unet_swin_unet.png"
	fig.suptitle("Confronto architetture: U-Net, Attention U-Net, Swin-Unet", fontsize=16)
	fig.savefig(out_loss, dpi=200, transparent=False)
	print(f"Salvato: {out_loss}")

	# Figura: IoU (validazione)
	fig2, ax2 = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
	style_axes(fig2, [ax2])
	for i, (lbl, *_rest) in enumerate(series):
		_, _, _, _, _, e_vi, vi = series[i]
		if vi:
			ax2.plot(e_vi, vi, label=lbl, color=palette[i % len(palette)])
	ax2.set_xlabel("Epoch", fontsize=12)
	ax2.set_ylabel("IoU", fontsize=12)
	ax2.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_iou = out_dir / "compare_iou_attention_unet_swin_unet.png"
	fig2.suptitle("Confronto IoU (val): U-Net, Attention U-Net, Swin-Unet", fontsize=16)
	fig2.savefig(out_iou, dpi=200, transparent=False)
	print(f"Salvato: {out_iou}")


if __name__ == "__main__":
	main()
