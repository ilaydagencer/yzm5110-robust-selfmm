
from pathlib import Path
import pickle

# Projenin ana klasorunu bul.
# YZM5110/MMSA/research/inspect_mosi.py
# -> YZM5110/
project_root = Path(__file__).resolve().parents[2]

# Veri dosyasinin konumu
data_path = (
    project_root
    / "data"
    / "MOSI"
    / "Processed"
    / "unaligned_50.pkl"
)

print("VERI_DOSYASI:", data_path, flush=True)

if not data_path.is_file():
    raise FileNotFoundError(
        f"Veri dosyasi bulunamadi: {data_path}"
    )

print("DOSYA_OKUNUYOR...", flush=True)

# Sadece resmi kaynaktan indirilmis ve
# SHA-256 ozeti dogrulanmis dosyayi ac.
with open(data_path, "rb") as f:
    data = pickle.load(f)

print("VERI_TIPI:", type(data).__name__)

if not isinstance(data, dict):
    raise TypeError("Beklenen veri yapisi dict degil.")

print("UST_ANAHTARLAR:", list(data.keys()))


def bilgi(value):
    """Bir alanin tipini ve boyutunu goster."""
    tip = type(value).__name__

    if hasattr(value, "shape"):
        return f"tip={tip}, shape={value.shape}"

    if isinstance(value, (list, tuple)):
        first_shape = (
            getattr(value[0], "shape", None)
            if len(value) > 0 else None
        )
        return (
            f"tip={tip}, uzunluk={len(value)}, "
            f"ilk_ornek_shape={first_shape}"
        )

    if isinstance(value, dict):
        return f"tip=dict, anahtarlar={list(value.keys())[:10]}"

    return f"tip={tip}"


for split in ["train", "valid", "test"]:
    print(f"\n===== {split.upper()} =====")

    if split not in data:
        print("BU_BOLUM_BULUNAMADI")
        continue

    part = data[split]

    if not isinstance(part, dict):
        print("BOLUM_TIPI:", type(part).__name__)
        continue

    print("ALANLAR:", list(part.keys()))

    for name, value in part.items():
        print(f"{name}: {bilgi(value)}")

    for candidate in [
        "id", "text", "audio",
        "vision", "regression_labels", "labels"
    ]:
        if candidate in part:
            try:
                print(
                    "ORNEK_SAYISI:",
                    len(part[candidate])
                )
                break
            except TypeError:
                pass

print("\nVERI_INCELEME_TAMAMLANDI")
