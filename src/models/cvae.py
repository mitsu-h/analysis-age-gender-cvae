import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import mlflow.pytorch
from mlflow import MlflowClient

class CVAE(pl.LightningModule):
    def __init__(self, input_channels, img_size, latent_dim, conditional_dim):
        super(CVAE, self).__init__()
        self.img_size = img_size
        # エンコーダの定義
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels + conditional_dim, 32, kernel_size=4, stride=2, padding=1), # 出力サイズの計算も重要
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(), # 畳み込み層の後にフラット化
        )

        # エンコーダの最後の層の出力サイズを計算
        self.enc_out_dim = self._calculate_conv_output_size()

        # 線形層
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, latent_dim)

        # デコーダの定義
        self.decoder_input = nn.Linear(latent_dim + conditional_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self._calculate_conv_transpose_input_size(), self._calculate_conv_transpose_input_size())),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def _calculate_conv_output_size(self):
        # エンコーダの出力サイズを計算
        size = self.img_size
        for _ in range(4):  # 4層の畳み込み層を通過する
            size = (size - 1) // 2 + 1  # 畳み込み層でのサイズ変化の計算
        return 256 * size * size  # チャネル数 * 縦 * 横

    def _calculate_conv_transpose_input_size(self):
        # 転置畳み込み層の入力サイズを計算
        size = self.img_size
        for _ in range(4):  # 4層の畳み込み層を通過する
            size = (size - 1) // 2 + 1
        return size

    def encode(self, x, c):
        # エンコーダへの入力
        c = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.img_size, self.img_size)
        inputs = torch.cat([x, c], dim=1)
        latent = self.encoder(inputs)
        return self.fc_mu(latent), self.fc_var(latent)

    def reparameterize(self, mean, log_var):
        # 再パラメータ化トリック
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, c):
        # デコーダへの入力
        inputs = torch.cat([z, c], dim=1)
        inputs = self.decoder_input(inputs)
        return self.decoder(inputs)

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var

    def training_step(self, batch, batch_idx):
        # トレーニングステップの定義
        x = batch["image"]
        a = batch["age"]
        g = batch["gender"]
        c = torch.stack([a, g], dim=1)

        recon_x, mean, log_var = self.forward(x, c)
        # 損失関数の計算
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + kl_div

        # ログ出力
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_div', kl_div)

        return loss

    def configure_optimizers(self):
        # オプティマイザの設定
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
