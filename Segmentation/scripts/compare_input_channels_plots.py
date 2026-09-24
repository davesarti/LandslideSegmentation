import json
import os
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
	train_loss = data.get("train_loss", [])
	val_loss = data.get("val_loss", [])
	# Nuovi campi per IoU con fallback su più nomi possibili
	train_iou = get_first_key(data, ["train_iou", "train_IoU", "train_mIoU", "miou_train", "iou_train"], [])
	val_iou   = get_first_key(data, ["val_iou",   "val_IoU",   "val_mIoU",   "miou_val",   "iou_val"], [])
	prefix = data.get("prefix", Path(json_path).stem.replace("_metrics", ""))
	return prefix, train_loss, val_loss, train_iou, val_iou


def main():
	root = Path(__file__).resolve().parent.parent  # repo root
	parser = argparse.ArgumentParser(description="Confronta gli input a 8 e 12 canali.")
	artifacts_dir = root / "artifacts"
	parser.add_argument("--metrics-dir", type=Path, default=artifacts_dir / "metrics", help="Directory dei JSON delle metriche.")
	parser.add_argument("--output-dir", type=Path, default=artifacts_dir / "plots", help="Directory in cui salvare i plot.")
	args = parser.parse_args()
	metrics_dir = args.metrics_dir
	out_dir = args.output_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	# Input metric files
	m1_path = metrics_dir / "unet_bce_metrics.json"
	m2_path = metrics_dir / "unet_bce_slope_ndvi_metrics.json"

	if not m1_path.exists():
		raise FileNotFoundError(f"File non trovato: {m1_path}")
	if not m2_path.exists():
		raise FileNotFoundError(f"File non trovato: {m2_path}")

	p1, tr1, vl1, ti1, vi1 = load_metrics(m1_path)
	p2, tr2, vl2, ti2, vi2 = load_metrics(m2_path)

	# Epoche (1-based)
	e1_tr = list(range(1, len(tr1) + 1))
	e2_tr = list(range(1, len(tr2) + 1))
	e1_vl = list(range(1, len(vl1) + 1))
	e2_vl = list(range(1, len(vl2) + 1))
	# Epoche per IoU
	e1_ti = list(range(1, len(ti1) + 1))
	e2_ti = list(range(1, len(ti2) + 1))
	e1_vi = list(range(1, len(vi1) + 1))
	e2_vi = list(range(1, len(vi2) + 1))

	# Figura con due subplot affiancati
	fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

	# Sfondo bianco e spines rimosse, aggiungi griglia (major e minor)
	fig.set_facecolor("white")
	fig.patch.set_alpha(1)
	for ax in axes:
		ax.set_facecolor("white")
		ax.minorticks_off()  # disattiva minor ticks
		ax.grid(True, which="major", linestyle="-", alpha=0.5)   # griglia leggermente più scura
		ax.grid(True, which="minor", linestyle=":", alpha=0.35)  # griglia leggermente più scura
		ax.tick_params(axis="both", which="both", length=0, labelsize=12)  # rimuove le tacchette e ingrandisce font
		for spine in ax.spines.values():
			spine.set_visible(False)

	# Train loss
	ax = axes[0]
	ax.plot(e1_tr, tr1, label="subset_8ch train", color="#1f77b4")
	ax.plot(e2_tr, tr2, label="full_12ch train", color="#ff7f0e")
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	# Val loss
	ax = axes[1]
	ax.plot(e1_vl, vl1, label="subset_8ch val", color="#1f77b4")
	ax.plot(e2_vl, vl2, label="full_12ch val", color="#ff7f0e")
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Loss", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_path = out_dir / "compare_train_val_loss_unet_bce_vs_slope_ndvi.png"
	fig.suptitle("Confronto loss: subset_8ch vs full_12ch", fontsize=16)
	fig.savefig(out_path, dpi=200, transparent=False)
	print(f"Salvato: {out_path}")

	# Grafico unico: IoU (solo due linee, validazione)
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

	ax = ax2
	ax.plot(e1_vi, vi1, label="subset_8ch", color="#1f77b4")
	ax.plot(e2_vi, vi2, label="full_12ch", color="#ff7f0e")
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("IoU", fontsize=12)
	ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="#cccccc", prop={"size": 12})

	out_path_iou = out_dir / "compare_iou_subset_8ch_vs_full_12ch.png"
	fig2.suptitle("Confronto IoU (val): subset_8ch vs full_12ch", fontsize=16)
	fig2.savefig(out_path_iou, dpi=200, transparent=False)
	print(f"Salvato: {out_path_iou}")


if __name__ == "__main__":
	main()

