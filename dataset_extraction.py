import os
from datasets import load_dataset

# cria diretório de saída
os.makedirs("dataset", exist_ok=True)

# carrega parquet local
data_files = {
    "train": [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ],
    "test": [
        "test-00000-of-00001.parquet",
    ],
}

ds = load_dataset("parquet", data_files=data_files)

img_id = 0

for split in ds:
    for sample in ds[split]:
        for img in sample["images"]:
            img.save(f"dataset/img_{img_id}.png")
            img_id += 1