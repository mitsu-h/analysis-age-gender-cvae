import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import mlflow

from src.models.cvae import CVAE  # 先ほど定義したモデルクラス
from src.data.lagenda import LagendaDataset  # データセットクラス
from src.data.transform import get_transforms  # データ変換の関数
from src.utils.model_init import init_weights_xavier  # モデルの初期化関数

torch.set_default_dtype(torch.float32)

def main(args):
    # データ変換の設定
    transform = get_transforms()

    # データセットとデータローダーの設定
    dataset = LagendaDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=11, persistent_workers=True)

    # モデルのインスタンス化
    model = CVAE(args.input_channels, args.img_size, args.latent_dim, args.conditional_dim)
    model.to(dtype=torch.float32)
    model = init_weights_xavier(model)

    # トレーナーの設定とトレーニングの実行
    trainer = Trainer(max_epochs=args.num_epochs, accelerator='mps', devices=1)
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(model, dataloader)

    # モデルの保存
    torch.save(model.state_dict(), args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ハイパーパラメータの設定
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--conditional_dim', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--csv_file', type=str, default='data/lagenda_annotation.csv')
    parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--save_model_path', type=str, default='trained_model.pth')

    args = parser.parse_args()
    main(args)
