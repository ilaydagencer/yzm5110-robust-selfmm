
import sentencepiece

from MMSA import MMSA_run


def main():
    print("LMF_EGITIM_BASLIYOR", flush=True)

    MMSA_run(
        "lmf",
        "mosi",
        seeds=[1111],
        gpu_ids=[0]
    )

    print("LMF_EGITIM_TAMAMLANDI", flush=True)


if __name__ == "__main__":
    main()
