"""Genera metadata.csv dai file raw della competizione Kaggle FREUID.

Da eseguire una volta dopo aver scaricato il dataset.
Produce data/freuid/metadata.csv con path assoluti per questa macchina.

Struttura attesa in --data-dir:
    train_labels.csv          ← fornito da Kaggle
    train/train/*.jpeg        ← immagini di training
    public_test/public_test/  ← immagini di test pubblico (opzionale)

Uso:
    uv run python scripts/prepare_metadata.py --data-dir data/freuid
    uv run python scripts/prepare_metadata.py --data-dir /content/drive/MyDrive/freuid_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


_LANGUAGE_MAP = {
    "EGYPT/DL":      "arabic",
    "BENIN/DL":      "latin",
    "GUINEA/DL":     "latin",
    "MAURITIUS/ID":  "latin",
    "MOZAMBIQUE/DL": "latin",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", required=True,
        help="Cartella con train_labels.csv e le immagini (es. data/freuid)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    labels_csv = data_dir / "train_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"train_labels.csv non trovato in {data_dir}\n"
            "Scarica il dataset da Kaggle e posizionalo in questa cartella."
        )

    df = pd.read_csv(labels_csv)
    print(f"Letto {labels_csv}: {len(df)} righe")

    # Path assoluto per questa macchina
    # train_labels.csv ha image_path come "train/xxxx.jpeg"
    # l'immagine si trova in data_dir/train/train/xxxx.jpeg
    def resolve_path(rel: str) -> str:
        # Prova prima il path diretto, poi con sottocartella duplicata
        p = data_dir / rel
        if p.exists():
            return str(p)
        stem = Path(rel).stem + Path(rel).suffix
        folder = Path(rel).parent.name
        p2 = data_dir / folder / folder / stem
        if p2.exists():
            return str(p2)
        return str(p)  # restituisce il path anche se non esiste (file_exists=False)

    df["image_path"] = df["image_path"].apply(resolve_path)
    df["file_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

    # Colonne derivate
    df["document_template"] = df["type"]
    df["language"] = df["type"].map(_LANGUAGE_MAP).fillna("unknown")
    df["attack_type"] = df.apply(
        lambda r: "bona_fide" if r["label"] == 0
        else ("digital_genai" if r["is_digital"] else "print_and_capture"),
        axis=1,
    )
    df["split"]  = "train"
    df["source"] = "freuid"

    # Colonne finali
    out = df[[
        "image_path", "label", "document_template", "language",
        "attack_type", "split", "source", "file_exists",
    ]]

    out_path = data_dir / "metadata.csv"
    out.to_csv(out_path, index=False)

    n_exists = int(out["file_exists"].sum())
    print(f"\nScritto: {out_path}")
    print(f"  Totale:       {len(out)}")
    print(f"  File trovati: {n_exists} / {len(out)}")
    print(f"  BF:           {(out['label']==0).sum()}")
    print(f"  Fraud:        {(out['label']==1).sum()}")
    if n_exists < len(out):
        print(f"\n  ATTENZIONE: {len(out)-n_exists} immagini non trovate.")
        print(f"  Controlla che --data-dir punti alla cartella corretta.")


if __name__ == "__main__":
    main()
