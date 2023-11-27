import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class LagendaDataset(Dataset):
    def __init__(self, csv_file, root_dir, crop_type='face', transform=None):
        """
        Args:
            csv_file (string): CSVファイルのパス。
            root_dir (string): 画像が格納されているディレクトリのパス。
            crop_type (string, optional): 画像から切り出す領域を指定する。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        self.root_dir = root_dir
        self.annotations = self._extract_age_gender_data(csv_file)
        self.crop_type = crop_type
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
        # faceの領域で顔を切り出す
        x0, y0, x1, y1 = self._select_crop_type(idx)
        image = image.crop((x0, y0, x1, y1))

        age = self.annotations.iloc[idx, 1]
        gender = self.annotations.iloc[idx, 2]

        if self.transform:
            image = self.transform(image).to(dtype=torch.float32)
            image /= 255.0

        sample = {'image': image, 'age': age, 'gender': gender}
        return sample
    
    def _select_crop_type(self, idx):
        """
        画像の中からバウンディングボックスで指定された領域を切り出す。
        """
        if self.crop_type == "face":
            return self.annotations.iloc[idx, 3], self.annotations.iloc[idx, 4], self.annotations.iloc[idx, 5], self.annotations.iloc[idx, 6]
        elif self.crop_type == "person":
            return self.annotations.iloc[idx, 7], self.annotations.iloc[idx, 8], self.annotations.iloc[idx, 9], self.annotations.iloc[idx, 10]
        else:
            raise ValueError("type must be 'face' or 'person'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='data/lagenda_annotation.csv')
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--crop_type', type=str, default='face')
    args = parser.parse_args()

    from transform import get_transforms
    transform = get_transforms()
    dataset = LagendaDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform=transform)

    print(dataset[0]["image"], dataset[0]["age"], dataset[0]["gender"])
