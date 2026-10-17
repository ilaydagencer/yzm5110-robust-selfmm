
import argparse
import csv
import time
from pathlib import Path

# Windows import uyumlulugu
import sentencepiece

import numpy as np
import torch

from easydict import EasyDict
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler

from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataLoader
from MMSA.models import AMIO


# ==================================================
# 1. PROJE AYARLARI
# ==================================================

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent

CONFIG = (
    PROJECT
    / "local_configs"
    / "selfmm_mosi_batch4.json"
)

CHECKPOINT = (
    PROJECT
    / "runs"
    / "selfmm_seed1111"
    / "saved_models"
    / "self_mm-mosi.pth"
)

OUTPUT_DIR = (
    REPO
    / "research"
    / "experiment_outputs"
)

# K0: Temiz
# K1: Ses eksik
# K2: Goruntu eksik
# K3: Ses ve goruntu eksik

CONDITIONS = ["K0", "K1", "K2", "K3"]


# ==================================================
# 2. MODELI VE VERIYI HAZIRLA
# ==================================================

def prepare_model_and_data(split):

    if not CONFIG.is_file():
        raise FileNotFoundError(
            f"Konfigurasyon bulunamadi: {CONFIG}"
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA bulunamadi."
        )

    # Tekrarlanabilirlik
    torch.manual_seed(1111)
    np.random.seed(1111)

    # Kaydedilen deney konfigurasyonunu oku
    args = get_config_regression(
        "self_mm",
        "mosi",
        config_file=str(CONFIG)
    )

    args = EasyDict(args)

    # Model bilgileri
    args.model_name = "self_mm"
    args.dataset_name = "mosi"
    args.train_mode = "regression"
    args.device = torch.device("cuda:0")

    # Standart CMU-MOSI ozellikleri
    args.custom_feature = None
    args.feature_T = None
    args.feature_A = None
    args.feature_V = None

    # Otomatik eksik veri uretimini kapat.
    # Bozulmalari evaluate() icinde uygulayacagiz.
    args.data_missing = False

    print("VERI:", args.featurePath)
    print("BATCH_SIZE:", args.batch_size)
    print("MODEL:", args.model_name)
    print("CHECKPOINT:", CHECKPOINT)

    # Train, valid ve test yukleyicileri
    
    # MMSA'nin orijinal veri yukleyicilerini olustur.
    loaders = MMDataLoader(
        args,
        num_workers=0
    )

    if split not in loaders:
        raise KeyError(
            f"Veri bolumu bulunamadi: {split}"
        )

    original_loader = loaders[split]

    # Validation ve test icin sabit sirali
    # yeni bir DataLoader olusturuyoruz.
    #
    # Orijinal veri seti degismiyor.
    # Yalnizca orneklerin okunma sirasi sabitleniyor.
    loader = DataLoader(
        dataset=original_loader.dataset,
        batch_size=original_loader.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=original_loader.collate_fn
    )

    # Kullanilan sampler'i kontrol et.
    print(
        "SAMPLER:",
        type(loader.sampler).__name__
    )

    print(
        "ORNEK_SAYISI:",
        len(loader.dataset)
    )

    if not isinstance(
        loader.sampler,
        SequentialSampler
    ):
        raise RuntimeError(
            "Sabit veri sirasi olusturulamadi."
        )

    print(
        "VERI_SIRASI_SABIT",
        flush=True
    )


    # Orijinal Self-MM modelini olustur
    model = AMIO(args)

    # Yalnizca kendi egittigimiz
    # guvenilir checkpoint yukleniyor.
    weights = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        weights,
        strict=True
    )

    model = model.to(args.device)
    model.eval()

    print(
        "MODEL_VE_VERI_HAZIR",
        flush=True
    )

    return model, loader, args.device


# ==================================================
# 3. TEK KOSULUN DEGERLENDIRILMESI
# ==================================================

def evaluate(model, loader, device, condition):

    predictions = []
    targets = []

    start_time = time.perf_counter()

    model.eval()

    with torch.no_grad():

        for batch in loader:

            # ------------------------------
            # Orijinal girdiler
            # ------------------------------

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

            # Dizi uzunluklarini degistirmiyoruz.
            audio_lengths = batch["audio_lengths"]
            vision_lengths = batch["vision_lengths"]

            # ------------------------------
            # Modalite eksikligi simulasyonu
            # ------------------------------

            # K1: Ses eksik
            # K3: Ses + goruntu eksik

            if condition in ("K1", "K3"):
                audio.zero_()

            # K2: Goruntu eksik
            # K3: Ses + goruntu eksik

            if condition in ("K2", "K3"):
                vision.zero_()

            # ------------------------------
            # Self-MM tahmini
            # ------------------------------

            outputs = model(
                text,
                (audio, audio_lengths),
                (vision, vision_lengths)
            )

            # M: Birlesik multimodal tahmin
            pred = (
                outputs["M"]
                .detach()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            # Gercek duygu etiketi
            true = (
                batch["labels"]["M"]
                .detach()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            predictions.extend(
                pred.tolist()
            )

            targets.extend(
                true.tolist()
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start_time
    )

    pred = np.asarray(
        predictions,
        dtype=np.float64
    )

    true = np.asarray(
        targets,
        dtype=np.float64
    )

    # ------------------------------
    # Sonuc kontrolu
    # ------------------------------

    if pred.size != true.size:
        raise RuntimeError(
            "Tahmin ve etiket sayilari farkli."
        )

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "Tahminlerde NaN veya Inf bulundu."
        )

    if not np.isfinite(true).all():
        raise RuntimeError(
            "Etiketlerde NaN veya Inf bulundu."
        )

    # ------------------------------
    # Accuracy ve F1 (Non0)
    # ------------------------------

    nonzero = true != 0

    if not np.any(nonzero):
        raise RuntimeError(
            "Non0 icin gecerli ornek bulunamadi."
        )

    binary_true = (
        true[nonzero] > 0
    ).astype(int)

    binary_pred = (
        pred[nonzero] > 0
    ).astype(int)

    accuracy = accuracy_score(
        binary_true,
        binary_pred
    )

    f1 = f1_score(
        binary_true,
        binary_pred,
        average="weighted",
        zero_division=0
    )

    # ------------------------------
    # MAE
    # ------------------------------

    mae = np.mean(
        np.abs(true - pred)
    )

    # ------------------------------
    # Correlation
    # ------------------------------

    if (
        np.std(true) > 0
        and np.std(pred) > 0
    ):
        corr = np.corrcoef(
            true,
            pred
        )[0, 1]

    else:
        corr = float("nan")

    # ------------------------------
    # Sonuclari hazirla
    # ------------------------------

    result = {
        "condition": condition,
        "samples": len(true),
        "nonzero_accuracy": round(
            float(accuracy), 6
        ),
        "nonzero_f1": round(
            float(f1), 6
        ),
        "mae": round(
            float(mae), 6
        ),
        "corr": round(
            float(corr), 6
        ),
        "time_seconds": round(
            elapsed, 3
        ),
    }

    # Metrikler + her ornegin tahmini
    return result, pred


# ==================================================
# 4. ANA DEGERLENDIRME
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid"
    )

    args_cli = parser.parse_args()

    split = args_cli.split

    print("=" * 50)
    print("SELF-MM ROBUSTNESS EVALUATION")
    print("SPLIT:", split)
    print("=" * 50, flush=True)

    # ------------------------------
    # Modeli ve veriyi yukle
    # ------------------------------

    model, loader, device = (
        prepare_model_and_data(split)
    )

    results = []

    # K0 tahminlerini saklayacagiz.
    baseline_predictions = None

    # ------------------------------
    # K0, K1, K2, K3 deneyleri
    # ------------------------------

    for condition in CONDITIONS:

        result, current_predictions = evaluate(
            model,
            loader,
            device,
            condition
        )

        # K0: Temiz baseline
        if condition == "K0":

            baseline_predictions = (
                current_predictions.copy()
            )

        if baseline_predictions is None:
            raise RuntimeError(
                "K0 baseline tahminleri bulunamadi."
            )

        # Ornek sayisi kontrolu
        if (
            current_predictions.shape
            != baseline_predictions.shape
        ):
            raise RuntimeError(
                "Tahmin dizilerinin boyutlari farkli."
            )

        # ------------------------------
        # Prediction Shift Analysis
        # ------------------------------

        delta = np.abs(
            current_predictions
            - baseline_predictions
        )

        # Ortalama tahmin farki
        mean_shift = float(
            delta.mean()
        )

        # En buyuk tahmin farki
        max_shift = float(
            delta.max()
        )

        # Pozitif / negatif tahmin
        # yonu degisen ornek sayisi
        sign_flips = int(
            np.sum(
                (current_predictions > 0)
                != (baseline_predictions > 0)
            )
        )

        # Sonuc sozlugune ekle
        result["mean_abs_pred_shift"] = round(
            mean_shift, 8
        )

        result["max_abs_pred_shift"] = round(
            max_shift, 8
        )

        result["prediction_sign_flips"] = (
            sign_flips
        )

        # ------------------------------
        # Konsol ciktisi
        # ------------------------------

        print(
            condition,
            "| Accuracy:",
            result["nonzero_accuracy"],
            "| F1:",
            result["nonzero_f1"],
            "| MAE:",
            result["mae"],
            "| Corr:",
            result["corr"],
            flush=True
        )

        print(
            "   Mean shift:",
            result["mean_abs_pred_shift"],
            "| Max shift:",
            result["max_abs_pred_shift"],
            "| Sign flips:",
            result["prediction_sign_flips"],
            flush=True
        )

        results.append(result)

        # ------------------------------
        # Test temiz baseline kontrolu
        # ------------------------------

        if (
            split == "test"
            and condition == "K0"
        ):

            acc_ok = abs(
                result["nonzero_accuracy"]
                - 0.8277
            ) < 0.005

            mae_ok = abs(
                result["mae"]
                - 0.7448
            ) < 0.01

            if not (acc_ok and mae_ok):

                raise RuntimeError(
                    "K0 sonucu onceki Self-MM "
                    "testiyle eslesmiyor. "
                    "Diger kosullara gecmeden "
                    "degerlendirme kodunu kontrol et."
                )

    # ==================================================
    # 5. SONUCLARI CSV OLARAK KAYDET
    # ==================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"selfmm_robustness_{split}_seed1111.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(results)

    print("=" * 50)

    print(
        "SONUCLAR_KAYDEDILDI:",
        output_path
    )

    print(
        "DEGERLENDIRME_TAMAMLANDI",
        flush=True
    )


# ==================================================
# 6. PROGRAM BASLANGICI
# ==================================================

if __name__ == "__main__":
    main()
