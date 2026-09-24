import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def get_first_key(d, keys, default=None):
	# Ritorna il primo valore disponibile tra più chiavi alternative
	for k in keys:
		if k in d and d[k] is not None:
			return d[k]
	return default


def load_metrics(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	train_loss = get_first_key(data, ["train_loss", "loss_train", "train"], [])
	val_loss = get_first_key(data, ["val_loss", "loss_val", "val"], [])
	# IoU (supporta più possibili nomi, come in plot.py)
	train_iou = get_first_key(data, ["train_iou", "train_IoU", "train_mIoU", "miou_train", "iou_train"], [])
	val_iou = get_first_key(data, ["val_iou", "val_IoU", "val_mIoU", "miou_val", "iou_val"], [])
	prefix = data.get("prefix", Path(json_path).stem.replace("_metrics", ""))
	return prefix, train_loss, val_loss, train_iou, val_iou


def main():
	# Percorsi
	root = Path(__file__).resolve().parent.parent  # repo root
	parser = argparse.ArgumentParser(description="Confronta i pesi della loss per Attention U-Net.")
	artifacts_dir = root / "artifacts"
	parser.add_argument("--metrics-dir", type=Path, default=artifacts_dir / "metrics", help="Directory dei JSON delle metriche.")
	parser.add_argument("--output-dir", type=Path, default=artifacts_dir / "plots", help="Directory in cui salvare i plot.")
	args = parser.parse_args()
	metrics_dir = args.metrics_dir
	out_dir = args.output_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	# File metriche richiesti
	paths = [
		metrics_dir / "attunet_bce_metrics.json",
		metrics_dir / "0.7attunet_bce_dice_metrics.json",
		metrics_dir / "0.5attunet_bce_dice_metrics.json",
		metrics_dir / "0.3attunet_bce_dice_metrics.json",
	]

	for p in paths:
		if not p.exists():
			raise FileNotFoundError(f"File non trovato: {p}")

	# Carica metriche
	series = []  # (label, e_train, train_vals, e_val, val_vals, e_tiou, tiou, e_viou, viou)
	for p in paths:
		label, tr, vl, ti, vi = load_metrics(p)
		# Epoche 1-based per ciascuna serie
		e_tr = list(range(1, len(tr) + 1))
		e_vl = list(range(1, len(vl) + 1))
		e_ti = list(range(1, len(ti) + 1)) if ti else []
		e_vi = list(range(1, len(vi) + 1)) if vi else []
		series.append((label, e_tr, tr, e_vl, vl, e_ti, ti, e_vi, vi))

	# Colori coerenti con palette base matplotlib
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

	# Figura con due subplot affiancati (Train e Val)
	fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

	# Stile coerente con plot.py
	fig.set_facecolor("white")
	fig.patch.set_alpha(1)
	for ax in axes:
		ax.set_facecolor("white")
		ax.minorticks_off()  # disattiva minor ticks
		ax.grid(True, which="major", linestyle="-", alpha=0.5)
		ax.grid(True, which="minor", linestyle=":", alpha=0.35)
		ax.tick_params(axis="both", which="both", length=0, labelsize=12)
		for spine in ax.spines.values():
			spine.set_visible(False)

	# Subplot: Train loss
	ax = axes[0]
	for i, (lbl, e_tr, tr, _, _, _, _, _, _) in enumerate(series):
		ax.plot(e_tr, tr, label=f"{lbl} train", color=colors[i % len(colors)])
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	# Subplot: Val loss
	ax = axes[1]
	for i, (lbl, _, _, e_vl, vl, _, _, _, _) in enumerate(series):
		ax.plot(e_vl, vl, label=f"{lbl} val", color=colors[i % len(colors)])
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_path = out_dir / "compare_train_val_loss_attention_unet_bce_and_dice_thresholds.png"
	fig.suptitle("Confronto loss: Attention U-Net BCE vs Attention U-Net BCE+Dice (0.7, 0.5, 0.3)", fontsize=16)
	fig.savefig(out_path, dpi=200, transparent=False)
	print(f"Salvato: {out_path}")

	# Grafico unico: IoU (validazione) per tutti i modelli
	fig2, ax2 = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
	fig2.set_facecolor("white")
	fig2.patch.set_alpha(1)
	ax2.set_facecolor("white")
	ax2.minorticks_off()
	ax2.grid(True, which="major", linestyle="-", alpha=0.5)
	ax2.grid(True, which="minor", linestyle=":", alpha=0.35)
	ax2.tick_params(axis="both", which="both", length=0, labelsize=12)
	for spine in ax2.spines.values():
		spine.set_visible(False)

	for i, (lbl, *_rest) in enumerate(series):
		# _rest structure: e_tr, tr, e_vl, vl, e_ti, ti, e_vi, vi
		_, _, e_vl, vl, e_ti, ti, e_vi, vi = _rest
		if vi:
			ax2.plot(e_vi, vi, label=f"{lbl}", color=colors[i % len(colors)])

	ax2.set_xlabel("Epoch", fontsize=12)
	ax2.set_ylabel("IoU", fontsize=12)
	ax2.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_path_iou = out_dir / "compare_iou_attention_unet_bce_and_dice_thresholds.png"
	fig2.suptitle("Confronto IoU (val): Attention U-Net BCE vs Attention U-Net BCE+Dice (0.7, 0.5, 0.3)", fontsize=16)
	fig2.savefig(out_path_iou, dpi=200, transparent=False)
	print(f"Salvato: {out_path_iou}")


if __name__ == "__main__":
	main()

