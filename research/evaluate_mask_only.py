
import csv
from pathlib import Path

# Windows import uyumlulugu
import sentencepiece

import numpy as np
import torch

from eval_robust import prepare_model_and_data
from evaluate_consistency import calculate_metrics
from probe_consistency_fusion import (
    find_fusion_layer,
    temporary_fusion_weights,
)


CONDITIONS = ["K0", "K1", "K2", "K3"]

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "experiment_outputs"
    / "mask_only_valid_seed1111.csv"
)


def evaluate_condition(
    model,
    loader,
    device,
    fusion_layer,
    condition
):
    baseline_preds = []
    masked_preds = []
    labels = []

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

            # Onceki deneylerle ayni giris bozulmasi.
            if condition in ("K1", "K3"):
                audio.zero_()

            if condition in ("K2", "K3"):
                vision.zero_()

            def forward():
                return model(
                    text,
                    (audio, audio_lengths),
                    (vision, vision_lengths)
                )

            # Referans: Orijinal Self-MM
            baseline = forward()

            # Mask-only: Mevcut modalite = 1
            #            Eksik modalite = 0
            mask = torch.ones(
                (text.shape[0], 3),
                dtype=audio.dtype,
                device=device
            )

            # Sutun sirasi: Text, Audio, Vision
            if condition in ("K1", "K3"):
                mask[:, 1] = 0.0

            if condition in ("K2", "K3"):
                mask[:, 2] = 0.0

            # Yalnizca fusion girdisine maske uygula.
            with temporary_fusion_weights(
                fusion_layer,
                mask
            ):
                masked = forward()

            baseline_preds.append(
                baseline["M"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            masked_preds.append(
                masked["M"]
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

    baseline_preds = np.concatenate(
        baseline_preds
    )

    masked_preds = np.concatenate(
        masked_preds
    )

    labels = np.concatenate(
        labels
    )

    if len(labels) != 229:
        raise RuntimeError(
            f"Beklenmeyen ornek sayisi: {len(labels)}"
        )

    if not (
        np.isfinite(baseline_preds).all()
        and np.isfinite(masked_preds).all()
        and np.isfinite(labels).all()
    ):
        raise RuntimeError(
            "Tahminlerde veya etiketlerde gecersiz deger."
        )

    baseline_metrics = calculate_metrics(
        baseline_preds,
        labels
    )

    masked_metrics = calculate_metrics(
        masked_preds,
        labels
    )

    shift = np.abs(
        baseline_preds - masked_preds
    )

    sign_flips = int(
        np.sum(
            (baseline_preds > 0)
            != (masked_preds > 0)
        )
    )

    # K0'da maskenin hicbir seyi degistirmemesi
    # gerekiyor. Bu kritik kontrolumuz.
    if condition == "K0":
        if float(shift.max()) > 1e-5:
            raise RuntimeError(
                "K0 identity kontrolu basarisiz."
            )

        if (
            abs(baseline_metrics["mae"] - 0.718312)
            > 0.001
        ):
            raise RuntimeError(
                "K0 baseline onceki deneyle eslesmiyor."
            )

        print("K0_IDENTITY_KONTROLU_BASARILI")

    return (
        baseline_metrics,
        masked_metrics,
        float(shift.mean()),
        float(shift.max()),
        sign_flips
    )


def main():
    print("=" * 55)
    print("SELF-MM MASK-ONLY ABLATION")
    print("SPLIT: VALID")
    print("=" * 55)

    model, loader, device = prepare_model_and_data(
        "valid"
    )

    fusion_layer = find_fusion_layer(
        model
    )

    rows = []

    for condition in CONDITIONS:

        (
            baseline,
            masked,
            mean_shift,
            max_shift,
            sign_flips
        ) = evaluate_condition(
            model,
            loader,
            device,
            fusion_layer,
            condition
        )

        print("\nCONDITION:", condition)

        print(
            "BASELINE_MAE:",
            round(baseline["mae"], 6)
        )

        print(
            "MASK_ONLY_MAE:",
            round(masked["mae"], 6)
        )

        print(
            "MAE_DELTA:",
            round(
                masked["mae"] - baseline["mae"],
                6
            )
        )

        print(
            "MASK_ONLY_ACCURACY:",
            round(masked["accuracy"], 6)
        )

        print(
            "MEAN_PREDICTION_SHIFT:",
            round(mean_shift, 6)
        )

        print(
            "MAX_PREDICTION_SHIFT:",
            round(max_shift, 6)
        )

        print(
            "SIGN_FLIPS:",
            sign_flips
        )

        for method, metrics in (
            ("baseline", baseline),
            ("mask_only", masked)
        ):
            rows.append({
                "split": "valid",
                "condition": condition,
                "method": method,
                "samples": 229,
                **metrics
            })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
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

    print("\n" + "=" * 55)
    print("SONUC_DOSYASI:", OUTPUT_PATH)
    print("MASK_ONLY_ABLATION_TAMAMLANDI")


if __name__ == "__main__":
    main()
