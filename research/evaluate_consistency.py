
import argparse
import csv
from pathlib import Path

# Windows import uyumlulugu
import sentencepiece

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score

from eval_robust import prepare_model_and_data
from probe_consistency_fusion import (
    find_fusion_layer,
    compute_consistency_weights,
    temporary_fusion_weights,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "experiment_outputs"
)


def calculate_metrics(pred, true):
    """CMU-MOSI regresyon ve Non0 metrikleri."""

    nonzero = true != 0

    binary_true = (true[nonzero] > 0).astype(int)
    binary_pred = (pred[nonzero] > 0).astype(int)

    accuracy = accuracy_score(
        binary_true, binary_pred
    )

    f1 = f1_score(
        binary_true,
        binary_pred,
        average="weighted",
        zero_division=0
    )

    mae = np.mean(np.abs(true - pred))

    if np.std(pred) > 0 and np.std(true) > 0:
        corr = np.corrcoef(true, pred)[0, 1]
    else:
        corr = float("nan")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "mae": float(mae),
        "corr": float(corr),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tau",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--condition",
        choices=["K0", "K1", "K2", "K3"],
        default="K0"
    )

    args = parser.parse_args()

    if args.tau <= 0:
        parser.error("Tau sifirdan buyuk olmali.")

    print("=" * 55)
    print("SELF-MM CONSISTENCY EVALUATION")
    print("SPLIT: VALID")
    print("CONDITION:", args.condition)
    print("TEMPERATURE:", args.tau)
    print("=" * 55)

    model, loader, device = prepare_model_and_data(
        "valid"
    )

    fusion_layer = find_fusion_layer(model)

    baseline_preds = []
    weighted_preds = []
    labels = []
    all_weights = []

    model.eval()

    with torch.inference_mode():

        for batch in loader:

            text = batch["text"].to(device)

            audio = (
                batch["audio"]
                .to(device)
                .clone()
            )

            vision = (
                batch["vision"]
                .to(device)
                .clone()
            )

            audio_lengths = batch["audio_lengths"]
            vision_lengths = batch["vision_lengths"]

            # Eksik modalite simulasyonu
            if args.condition in ("K1", "K3"):
                audio.zero_()

            if args.condition in ("K2", "K3"):
                vision.zero_()

            def forward():
                return model(
                    text,
                    (audio, audio_lengths),
                    (vision, vision_lengths)
                )

            # 1. Orijinal Self-MM
            baseline = forward()

            # 2. Tek-modlu tahminlerden agirliklar
            weights, _ = compute_consistency_weights(
                baseline,
                args.tau
            )

            # Softmax agirlik toplami 3 olmali
            weight_sums = weights.sum(dim=1)

            if not torch.allclose(
                weight_sums,
                torch.full_like(weight_sums, 3.0),
                atol=1e-5
            ):
                raise RuntimeError(
                    "Agirlik toplami beklenen degerde degil."
                )

            # 3. Agirlikli fusion
            with temporary_fusion_weights(
                fusion_layer,
                weights
            ):
                weighted = forward()

            baseline_preds.append(
                baseline["M"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            weighted_preds.append(
                weighted["M"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            labels.append(
                batch["labels"]["M"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            all_weights.append(
                weights.cpu().numpy()
            )

    # Batch sonuclarini birlestir
    baseline_preds = np.concatenate(baseline_preds)
    weighted_preds = np.concatenate(weighted_preds)
    labels = np.concatenate(labels)
    all_weights = np.concatenate(all_weights)

    if len(labels) != 229:
        raise RuntimeError(
            f"Beklenmeyen validation sayisi: {len(labels)}"
        )

    if not (
        np.isfinite(baseline_preds).all()
        and np.isfinite(weighted_preds).all()
    ):
        raise RuntimeError("Gecersiz tahmin degeri.")

    baseline_metrics = calculate_metrics(
        baseline_preds, labels
    )

    weighted_metrics = calculate_metrics(
        weighted_preds, labels
    )

    # Temiz baseline dogrulamasi
    if args.condition == "K0":

        if (
            abs(baseline_metrics["accuracy"] - 0.856481)
            > 0.001
            or abs(baseline_metrics["mae"] - 0.718312)
            > 0.001
        ):
            raise RuntimeError(
                "Temiz baseline onceki sonucla eslesmiyor."
            )

    # Tahmin farklari
    differences = np.abs(
        weighted_preds - baseline_preds
    )

    sign_flips = np.sum(
        (weighted_preds > 0)
        != (baseline_preds > 0)
    )

    mean_weights = np.mean(
        all_weights,
        axis=0
    )

    print("\nBASELINE:")
    print(baseline_metrics)

    print("\nCONSISTENCY WEIGHTED:")
    print(weighted_metrics)

    print("\nTAHMIN FARKLARI")
    print(
        "MEAN_SHIFT:",
        round(float(differences.mean()), 6)
    )
    print(
        "MAX_SHIFT:",
        round(float(differences.max()), 6)
    )
    print(
        "SIGN_FLIPS:",
        int(sign_flips)
    )

    print("\nORTALAMA AGIRLIKLAR")
    print("TEXT:", round(float(mean_weights[0]), 6))
    print("AUDIO:", round(float(mean_weights[1]), 6))
    print("VISION:", round(float(mean_weights[2]), 6))

    # Sonuclari kaydet
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tau_name = f"{args.tau:g}".replace(".", "p")

    output_path = (
        OUTPUT_DIR
        / f"consistency_valid_{args.condition}_tau{tau_name}.csv"
    )

    rows = []

    for name, metrics in [
        ("baseline", baseline_metrics),
        ("consistency", weighted_metrics),
    ]:
        rows.append({
            "split": "valid",
            "condition": args.condition,
            "method": name,
            "tau": args.tau,
            "samples": len(labels),
            **metrics
        })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSONUC_DOSYASI:", output_path)
    print("CONSISTENCY_VALIDATION_TAMAMLANDI")


if __name__ == "__main__":
    main()
