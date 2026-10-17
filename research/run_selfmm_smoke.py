
import json
import sys
import time
from pathlib import Path

# Windows'taki onceki import hatasi icin
# sentencepiece, MMSA'dan once yukleniyor.
import sentencepiece
import torch

from MMSA import MMSA_run


# ==================================================
# 1. PROJE KLASORLERI
# ==================================================

# C:/Users/ilayda/YZM5110/MMSA
repo_dir = Path(__file__).resolve().parents[1]

# C:/Users/ilayda/YZM5110
project_dir = repo_dir.parent

# MMSA'nin orijinal konfigürasyonu
original_config = (
    repo_dir
    / "src"
    / "MMSA"
    / "config"
    / "config_regression.json"
)

# Deneyimize ozel konfigürasyon
local_config = (
    project_dir
    / "local_configs"
    / "selfmm_mosi_batch4.json"
)

# CMU-MOSI veri seti
data_file = (
    project_dir
    / "data"
    / "MOSI"
    / "Processed"
    / "unaligned_50.pkl"
)

# Egitim sonuclari
run_dir = (
    project_dir
    / "runs"
    / "selfmm_seed1111"
)


# ==================================================
# 2. GPU BELLEK OLCUMU
# ==================================================

def print_gpu_memory():
    """
    Egitim sonunda GPU bellek kullanimini olcer.

    Bellek olcumu hata verirse programi cokertmez.
    """

    if not torch.cuda.is_available():
        print("GPU_BELLEK_OLCUMU: CUDA kullanilamiyor")
        return

    try:
        torch.cuda.synchronize()

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

        print(
            "PEAK_GPU_ALLOCATED_MB:",
            round(peak_allocated, 2)
        )

        print(
            "PEAK_GPU_RESERVED_MB:",
            round(peak_reserved, 2)
        )

    except RuntimeError as e:
        print(
            "GPU_BELLEK_OLCUM_UYARISI:",
            str(e)
        )


# ==================================================
# 3. ANA PROGRAM
# ==================================================

def main():

    # ----------------------------------------------
    # Dosyalari kontrol et
    # ----------------------------------------------

    if not original_config.is_file():
        raise FileNotFoundError(
            f"Konfigürasyon bulunamadi: {original_config}"
        )

    if not data_file.is_file():
        raise FileNotFoundError(
            f"Veri dosyasi bulunamadi: {data_file}"
        )

    # ----------------------------------------------
    # Orijinal konfigürasyonu oku
    # ----------------------------------------------

    with original_config.open(
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    # ----------------------------------------------
    # Deney ayarlarini yap
    # ----------------------------------------------

    config["datasetCommonParams"]["dataset_root_dir"] = (
        (project_dir / "data").as_posix()
    )

    # 4 GB VRAM icin batch size 4
    config["self_mm"]["datasetParams"]["mosi"][
        "batch_size"
    ] = 4

    # BERT fine-tuning ayarini degistirmiyoruz.
    # Orijinal MMSA konfigürasyonu korunuyor.

    # ----------------------------------------------
    # Yerel konfigürasyonu kaydet
    # ----------------------------------------------

    local_config.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with local_config.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=2
        )

    # ----------------------------------------------
    # Deney bilgilerini yazdir
    # ----------------------------------------------

    print("=" * 50)
    print("YZM5110 - SELF-MM BASELINE EXPERIMENT")
    print("=" * 50)

    print("MODEL: Self-MM")
    print("DATASET: CMU-MOSI")

    print(
        "BATCH_SIZE:",
        config["self_mm"]["datasetParams"]["mosi"][
            "batch_size"
        ]
    )

    print(
        "BERT_FINETUNE:",
        config["self_mm"]["commonParams"][
            "use_finetune"
        ]
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    print("CONFIG:", local_config)
    print("DATA:", data_file)
    print("RESULT_DIRECTORY:", run_dir)

    print("=" * 50, flush=True)

    # ----------------------------------------------
    # Sadece konfigürasyon kontrolu
    # ----------------------------------------------

    if "--check" in sys.argv:
        print(
            "KONFIGURASYON_KONTROLU_BASARILI",
            flush=True
        )
        return

    # ----------------------------------------------
    # CUDA kontrolu
    # ----------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA bulunamadi. GPU egitimi baslatilamiyor."
        )

    # ----------------------------------------------
    # Sonuc klasorlerini olustur
    # ----------------------------------------------

    for name in [
        "saved_models",
        "results",
        "logs"
    ]:
        (run_dir / name).mkdir(
            parents=True,
            exist_ok=True
        )

    # ----------------------------------------------
    # Egitimi baslat
    # ----------------------------------------------

    start_time = time.perf_counter()

    print(
        "SELFMM_EGITIM_BASLIYOR",
        flush=True
    )

    # Burada orijinal MMSA Self-MM modeli egitilir.
    # Baseline mimarisinde degisiklik yapilmamistir.

    MMSA_run(
        "self_mm",
        "mosi",
        config_file=str(local_config),
        seeds=[1111],
        gpu_ids=[0],
        num_workers=0,
        model_save_dir=str(
            run_dir / "saved_models"
        ),
        res_save_dir=str(
            run_dir / "results"
        ),
        log_dir=str(
            run_dir / "logs"
        ),
    )

    # ----------------------------------------------
    # Egitim basariyla bitti
    # ----------------------------------------------

    elapsed = (
        time.perf_counter() - start_time
    )

    print("=" * 50)

    print(
        "SELFMM_EGITIM_TAMAMLANDI",
        flush=True
    )

    print(
        "SURE_SANIYE:",
        round(elapsed, 2)
    )

    print(
        "SURE_DAKIKA:",
        round(elapsed / 60, 2)
    )

    # ----------------------------------------------
    # GPU bellek olcumu
    # ----------------------------------------------

    print_gpu_memory()

    print("=" * 50)
    print("DENEY_SONUCLANDI", flush=True)


# ==================================================
# 4. PROGRAM BASLANGICI
# ==================================================

if __name__ == "__main__":
    main()
