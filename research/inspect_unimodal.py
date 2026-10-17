
import sentencepiece
import numpy as np
import torch

from eval_robust import prepare_model_and_data


def main():
    model, loader, device = prepare_model_and_data("valid")

    predictions = {
        "M": [],
        "T": [],
        "A": [],
        "V": []
    }

    labels = []

    model.eval()

    with torch.no_grad():
        for batch in loader:

            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)

            outputs = model(
                text,
                (audio, batch["audio_lengths"]),
                (vision, batch["vision_lengths"])
            )

            for key in predictions:
                values = (
                    outputs[key]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .numpy()
                )

                predictions[key].append(values)

            true = (
                batch["labels"]["M"]
                .detach()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            labels.append(true)

    labels = np.concatenate(labels)

    print("\nUNIMODAL PREDICTION ANALYSIS")
    print("SAMPLES:", len(labels))

    for key, batches in predictions.items():
        values = np.concatenate(batches)

        mean = np.mean(values)
        std = np.std(values)
        minimum = np.min(values)
        maximum = np.max(values)

        mae = np.mean(np.abs(values - labels))

        if std > 1e-10 and np.std(labels) > 1e-10:
            corr = np.corrcoef(values, labels)[0, 1]
        else:
            corr = float("nan")

        print(f"\nMODALITY: {key}")
        print("MEAN:", round(float(mean), 6))
        print("STD:", round(float(std), 6))
        print("MIN:", round(float(minimum), 6))
        print("MAX:", round(float(maximum), 6))
        print("MAE:", round(float(mae), 6))
        print("CORR:", round(float(corr), 6))

    print("\nANALIZ_TAMAMLANDI")


if __name__ == "__main__":
    main()
