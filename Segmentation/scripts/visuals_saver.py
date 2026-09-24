
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from data_utils import SegmentationSingleDataset

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Scala un'immagine float/np in range 0..1 o arbitrario a uint8 0..255 per plotting."""
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx > mn:
        x = (x - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def _extract_pre_post_rgb_nir(data_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estrae pre/post RGB e NIR assumendo layout [pre R,G,B,NIR, post R,G,B,NIR].
    Ricade su heursitiche se i canali non sono 8.
    Ritorna: pre_rgb(H,W,3), post_rgb(H,W,3), pre_nir(H,W), post_nir(H,W)
    """
    c, h, w = data_np.shape
    # Pre RGB
    if c >= 3:
        pre_rgb = data_np[:3].transpose(1, 2, 0)
    else:
        pre_rgb = np.repeat(data_np[0:1].transpose(1, 2, 0), 3, axis=2)
    # Post RGB
    if c >= 7:
        post_rgb = data_np[4:7].transpose(1, 2, 0)
    elif c >= 3:
        post_rgb = data_np[:3].transpose(1, 2, 0)
    else:
        post_rgb = np.repeat(data_np[-1:].transpose(1, 2, 0), 3, axis=2)
    # NIR
    pre_nir = data_np[3] if c >= 4 else data_np[min(c - 1, 0)]
    post_nir = data_np[7] if c >= 8 else (data_np[3] if c >= 4 else data_np[min(c - 1, 0)])
    return pre_rgb, post_rgb, pre_nir, post_nir


def _overlay_on_rgb(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """Sovrappone una maschera binaria su un RGB uint8 con colore e alpha dati."""
    base = _to_uint8(rgb)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    out = base.copy()
    mask_bool = mask.astype(bool)
    color_arr = np.zeros_like(out)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    out[mask_bool] = (
        (1 - alpha) * out[mask_bool].astype(np.float32) + alpha * color_arr[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return out


def _overlay_pair_on_rgb(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    color_gt=(0, 255, 0),
    color_pred=(255, 0, 255),
    alpha: float = 0.45,
    both_color=(255, 255, 255),
    both_alpha: float = 0.6,
) -> np.ndarray:
    """Sovrappone GT e Prediction su un RGB.
    - GT in color_gt
    - Pred in color_pred
    - Intersezione (GT ∩ Pred) in both_color (bianco default) per massima leggibilità.
    """
    base = _to_uint8(rgb)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    out = base.copy()

    gt_b = gt_mask.astype(bool)
    pd_b = pred_mask.astype(bool)
    both = gt_b & pd_b
    gt_only = gt_b & ~both
    pd_only = pd_b & ~both

    # Prepara array colore
    color_arr_gt = np.zeros_like(out)
    color_arr_gt[..., 0] = color_gt[0]
    color_arr_gt[..., 1] = color_gt[1]
    color_arr_gt[..., 2] = color_gt[2]

    color_arr_pd = np.zeros_like(out)
    color_arr_pd[..., 0] = color_pred[0]
    color_arr_pd[..., 1] = color_pred[1]
    color_arr_pd[..., 2] = color_pred[2]

    color_arr_both = np.zeros_like(out)
    color_arr_both[..., 0] = both_color[0]
    color_arr_both[..., 1] = both_color[1]
    color_arr_both[..., 2] = both_color[2]

    # Applica blending per ciascuna regione
    out[gt_only] = (
        (1 - alpha) * out[gt_only].astype(np.float32) + alpha * color_arr_gt[gt_only].astype(np.float32)
    ).astype(np.uint8)

    out[pd_only] = (
        (1 - alpha) * out[pd_only].astype(np.float32) + alpha * color_arr_pd[pd_only].astype(np.float32)
    ).astype(np.uint8)

    out[both] = (
        (1 - both_alpha) * out[both].astype(np.float32) + both_alpha * color_arr_both[both].astype(np.float32)
    ).astype(np.uint8)

    return out


def save_matplotlib_visuals(
    model: nn.Module,
    dataset: SegmentationSingleDataset,
    device: torch.device,
    out_dir: str | Path,
    num_samples: int = 5,
    threshold: float = 0.5,
) -> None:
    """Salva per ogni patch due immagini:
    - Pannelli con pre/post RGB e NIR
    - Overlay: GT su post RGB, Pred su post RGB, GT+Pred su post RGB
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            data, landslide = dataset[i]
            data_batch = data.unsqueeze(0).to(device)

            pred = model(data_batch).squeeze(0).cpu()
            pred_prob = torch.sigmoid(pred)
            pred_mask = (pred_prob > threshold).float()

            data_np = data.cpu().numpy()
            gt_np = landslide.cpu().numpy().squeeze()
            pred_np = pred_mask.numpy().squeeze()

            pre_rgb, post_rgb, pre_nir, post_nir = _extract_pre_post_rgb_nir(data_np)

            # Figura A: pre/post RGB + NIR (ripristina titoli per pannello, nessun suptitle)
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
            axes = np.atleast_1d(axes)
            axes[0].imshow(_to_uint8(pre_rgb))
            axes[0].set_title("Pre RGB")
            axes[1].imshow(_to_uint8(post_rgb))
            axes[1].set_title("Post RGB")
            axes[2].imshow(_to_uint8(pre_nir), cmap="gray")
            axes[2].set_title("Pre NIR")
            axes[3].imshow(_to_uint8(post_nir), cmap="gray")
            axes[3].set_title("Post NIR")
            for ax in axes:
                ax.axis("off")
            # Niente suptitle; compattiamo ma lasciamo margini per i titoli
            fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.25)
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            fig.savefig(out_dir / f"sample_{i:03d}_a_rgb_nir.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Figura B: overlay
            post_uint8 = _to_uint8(post_rgb)
            gt_overlay = _overlay_on_rgb(post_uint8, gt_np, color=(0, 255, 0), alpha=0.45)
            pred_overlay = _overlay_on_rgb(post_uint8, pred_np, color=(255, 0, 255), alpha=0.45)
            both_overlay = _overlay_pair_on_rgb(post_uint8, gt_np, pred_np, color_gt=(0, 255, 0), color_pred=(255, 0, 255), both_color=(255, 255, 255))

            fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.4))
            axes2[0].imshow(gt_overlay)
            axes2[0].set_title("Ground Truth")
            axes2[1].imshow(pred_overlay)
            axes2[1].set_title("Predizione")
            axes2[2].imshow(both_overlay)
            axes2[2].set_title("Intersezione (GT ∩ Pred)")
            for ax in axes2:
                ax.axis("off")
            # Layout compatto identico alla figura RGB+NIR per stessa spaziatura
            fig2.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            fig2.savefig(out_dir / f"sample_{i:03d}_b_overlays.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
