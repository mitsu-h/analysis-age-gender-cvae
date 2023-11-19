import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class LagendaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): CSVファイルのパス。
            root_dir (string): 画像が格納されているディレクトリのパス。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        self.root_dir = root_dir
        self.annotations = self._extract_age_gender_data(csv_file)
        self.transform = transform

    def _extract_age_gender_data(self, csv_file):
        annotations = pd.read_csv(csv_file)

        # 年齢、性別が欠損値でないデータのみ抽出
        annotations = annotations[annotations['age'] >= 0]
        annotations = annotations[annotations['age'] <= 100]
        annotations = annotations[annotations['gender'] != -1]

        # 年齢をfloat型に変換
        annotations['age'] = annotations['age'].astype(np.float32) / 100

        # 性別がM, Fと定義されているので、0, 1に変換
        annotations['gender'] = annotations['gender'].replace({'M': 0, 'F': 1}).astype(np.float32)

        # pathを確認し、存在しないファイル名を削除
        annotations = annotations[annotations['img_name'].apply(lambda x: os.path.exists(os.path.join(self.root_dir, x)))]

        return annotations


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        age = self.annotations.iloc[idx, 1]
        gender = self.annotations.iloc[idx, 2]

        if self.transform:
            image = self.transform(image).to(dtype=torch.float32)
            image /= 255.0

        sample = {'image': image, 'age': age, 'gender': gender}
        return sample
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='data/lagenda_annotation.csv')
    parser.add_argument('--root_dir', type=str, default='data/')
    args = parser.parse_args()

    from transform import get_transforms
    transform = get_transforms()
    dataset = LagendaDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform=transform)

    print(dataset[0]["image"], dataset[0]["age"], dataset[0]["gender"])
