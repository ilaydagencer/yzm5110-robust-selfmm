
import argparse
from contextlib import contextmanager

# Windows import uyumlulugu
import sentencepiece

import torch

from eval_robust import prepare_model_and_data


# Self-MM'in kodlayici cikis boyutlari:
# text_out=768, audio_out=16, video_out=32
FEATURE_DIMS = (768, 16, 32)


def find_fusion_layer(model):
    """Orijinal Self-MM'in ilk fusion katmanini bul."""

    matches = []

    for name, module in model.named_modules():
        if (
            name.endswith("post_fusion_layer_1")
            and isinstance(module, torch.nn.Linear)
        ):
            matches.append((name, module))

    if len(matches) != 1:
        raise RuntimeError(
            f"Beklenen tek fusion katmani, bulunan: {matches}"
        )

    name, layer = matches[0]

    if layer.in_features != sum(FEATURE_DIMS):
        raise RuntimeError(
            "Fusion giris boyutlari uyusmuyor: "
            f"{layer.in_features}"
        )

    print("FUSION_KATMANI:", name)
    print("FUSION_GIRIS_BOYUTU:", layer.in_features)

    return layer


def compute_consistency_weights(outputs, temperature):
    """
    Tek-modlu tahminler arasindaki farktan
    uc modalite icin agirlik hesapla.
    """

    if temperature <= 0:
        raise ValueError("Temperature sifirdan buyuk olmali.")

    y_t = outputs["T"].reshape(-1)
    y_a = outputs["A"].reshape(-1)
    y_v = outputs["V"].reshape(-1)

    # Her modalitenin diger ikisinden sapmasi
    d_t = torch.abs(y_t - (y_a + y_v) / 2)
    d_a = torch.abs(y_a - (y_t + y_v) / 2)
    d_v = torch.abs(y_v - (y_t + y_a) / 2)

    deviations = torch.stack(
        [d_t, d_a, d_v],
        dim=1
    )

    # Ortalama agirlik 1 olacak bicimde
    # olcegi korumaya calisiyoruz.
    weights = 3.0 * torch.softmax(
        -deviations / temperature,
        dim=1
    )

    return weights, deviations


@contextmanager
def temporary_fusion_weights(layer, weights):
    """
    Fusion katmanina gecici hook ekler.
    Orijinal model parametrelerini degistirmez.
    """

    def hook(module, inputs):
        x = inputs[0]

        if x.ndim != 2:
            raise RuntimeError("Fusion girdisi 2D olmali.")

        if x.shape[0] != weights.shape[0]:
            raise RuntimeError("Batch boyutlari farkli.")

        text_h, audio_h, video_h = torch.split(
            x,
            FEATURE_DIMS,
            dim=-1
        )

        weighted_x = torch.cat(
            [
                text_h * weights[:, 0:1],
                audio_h * weights[:, 1:2],
                video_h * weights[:, 2:3]
            ],
            dim=-1
        )

        return (weighted_x,) + inputs[1:]

    handle = layer.register_forward_pre_hook(hook)

    try:
        yield
    finally:
        # Hata olsa bile hook kaldirilir.
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0
    )
    args = parser.parse_args()

    print("=" * 55)
    print("SELF-MM CONSISTENCY FUSION PROBE")
    print("TEMPERATURE:", args.tau)
    print("=" * 55)

    # Mevcut checkpoint ve validation verisi
    model, loader, device = prepare_model_and_data(
        "valid"
    )

    model.eval()

    fusion_layer = find_fusion_layer(model)

    # Yalnizca ilk batch: 4 ornek
    batch = next(iter(loader))

    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)

    audio_lengths = batch["audio_lengths"]
    vision_lengths = batch["vision_lengths"]

    def forward():
        return model(
            text,
            (audio, audio_lengths),
            (vision, vision_lengths)
        )

    with torch.inference_mode():

        # 1. Orijinal Self-MM
        baseline = forward()

        # 2. Tum agirliklar 1
        uniform_weights = torch.ones(
            text.shape[0],
            3,
            device=device
        )

        with temporary_fusion_weights(
            fusion_layer,
            uniform_weights
        ):
            uniform_output = forward()

        # 3. Esit agirlik kontrolu
        baseline_prediction = baseline["M"].reshape(-1)
        uniform_prediction = uniform_output["M"].reshape(-1)

        max_control_error = torch.max(
            torch.abs(
                baseline_prediction - uniform_prediction
            )
        ).item()

        print(
            "ESIT_AGIRLIK_KONTROL_FARKI:",
            round(max_control_error, 10)
        )

        if max_control_error > 1e-5:
            raise RuntimeError(
                "Esit agirlik kontrolu basarisiz. "
                "Orijinal fusion korunmuyor."
            )

        print("ESIT_AGIRLIK_KONTROLU_BASARILI")

        # 4. Tek-modlu tahminlere gore agirliklar
        weights, deviations = compute_consistency_weights(
            baseline,
            args.tau
        )

        # 5. Yeni agirlikli fusion tahmini
        with temporary_fusion_weights(
            fusion_layer,
            weights
        ):
            weighted_output = forward()

        weighted_prediction = (
            weighted_output["M"].reshape(-1)
        )

        # 6. Sonuclari goster
        print("\nORNEK BAZLI ANALIZ")

        for i in range(text.shape[0]):
            print(f"\nORNEK {i + 1}")

            print(
                "T A V:",
                [
                    round(
                        baseline[k][i].item(), 5
                    )
                    for k in ("T", "A", "V")
                ]
            )

            print(
                "SAPMALAR:",
                [
                    round(x, 5)
                    for x in deviations[i].tolist()
                ]
            )

            print(
                "AGIRLIKLAR:",
                [
                    round(x, 5)
                    for x in weights[i].tolist()
                ]
            )

            print(
                "ORIJINAL_TAHMIN:",
                round(baseline_prediction[i].item(), 6)
            )

            print(
                "AGIRLIKLI_TAHMIN:",
                round(weighted_prediction[i].item(), 6)
            )

        shift = torch.abs(
            weighted_prediction - baseline_prediction
        )

        print("\n" + "=" * 55)
        print("ORTALAMA_TAHMIN_FARKI:", round(shift.mean().item(), 6))
        print("EN_BUYUK_TAHMIN_FARKI:", round(shift.max().item(), 6))
        print("PROBE_TAMAMLANDI")


if __name__ == "__main__":
    main()
