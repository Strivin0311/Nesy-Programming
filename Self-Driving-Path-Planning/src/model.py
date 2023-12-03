import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)
from src.config import setup_cfg
from src.mingpt import GPTConfig, GPT


def build_model_resnet(cfg, device, num_layers=18, pretrained=False):
    model_weights_dict = {
        18: (resnet18, ResNet18_Weights.DEFAULT),
        34: (resnet34, ResNet34_Weights.DEFAULT),
        50: (resnet50, ResNet50_Weights.DEFAULT),
        101: (resnet101, ResNet101_Weights.DEFAULT),
    }

    # construct basic resnet model with pretrained weights or not
    weights = model_weights_dict[num_layers][1] if pretrained else None
    basic_model = model_weights_dict[num_layers][0](weights=weights)

    class ResModel(nn.Module):
        def __init__(self, basic_model, width, height):
            super(ResModel, self).__init__()
            self.basic_model = basic_model
            self.width = width
            self.height = height

            in_channels = self.basic_model._modules['fc'].in_features
            self.basic_model.fc = nn.Linear(in_features=in_channels,
                                          out_features=self.width * self.height,
                                          bias=True)
            self.activation = nn.Sigmoid()

        def forward(self, x):
            x = self.basic_model(x)
            x = x.view(x.shape[0], self.width, self.height)
            x = self.activation(x)

            return x

    model = ResModel(basic_model, cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)
    model.to(device)
    return model


def build_model_traj_resnet(cfg, device, num_layers=18, pretrained=False):
    model_weights_dict = {
        18: (resnet18, ResNet18_Weights.DEFAULT),
        34: (resnet34, ResNet34_Weights.DEFAULT),
        50: (resnet50, ResNet50_Weights.DEFAULT),
        101: (resnet101, ResNet101_Weights.DEFAULT),
    }

    # construct basic resnet model with pretrained weights or not
    weights = model_weights_dict[num_layers][1] if pretrained else None
    basic_model = model_weights_dict[num_layers][0](weights=weights)

    class PointEmbed(nn.Module):
        def __init__(self, input_size=10, emb_size=6*12):
            super().__init__()
            self.input_size = input_size
            self.emb_size = emb_size
            self.embedding_layer = nn.Linear(in_features=1, out_features=emb_size)

        def forward(self, p):
            """
            :param p: a matrix with shape=(input_size, input_size) where the point to encode is 1, the other points are 0
            :return: the embedding vector
            """
            p = p.view(-1, self.input_size * self.input_size, 1)  # shape=(batch_size, input_size**2, 1)
            p = self.embedding_layer(p)  # shape=(batch_size, input_size**2, emb_size)
            p = F.relu(p)

            return p

    class ResModel(nn.Module):
        def __init__(self, basic_model, cfg, width, height):
            super(ResModel, self).__init__()
            self.basic_model = nn.Sequential(*list(basic_model.children())[:-2])
            self.width = width
            self.height = height
            self.emb_size = 6 * 12
            self.config = cfg

            self.start_emb = PointEmbed(input_size=cfg.OUTPUT_WIDTH, emb_size=self.emb_size)
            self.target_emb = PointEmbed(input_size=cfg.OUTPUT_WIDTH, emb_size=self.emb_size)
            self.drop = nn.Dropout(0.5)

            self.conv_out = nn.Sequential(
                nn.Conv2d(512, 100, 3, 2, 1),
                nn.BatchNorm2d(100),
                nn.ReLU()
            )

            self.ln_f1 = nn.LayerNorm(self.emb_size)
            self.ln_f2 = nn.LayerNorm(self.emb_size)
            self.head1 = nn.Linear(self.emb_size, 2, bias=True)
            self.head2 = nn.Linear(self.emb_size, 2, bias=True)

        def forward(self, x, targets=None, sp=None, tp=None):
            x = self.basic_model(x)
            x = self.conv_out(x)
            x = x.view(x.shape[0], x.shape[1], 6*12)

            if sp is not None:
                start_embeddings = self.start_emb(sp)
                x = x + start_embeddings
            if tp is not None:
                target_embeddings = self.target_emb(tp)
                x = x + target_embeddings

            x = self.drop(x)

            x1 = self.head1(self.ln_f1(x)).view(x.shape[0], self.width, self.height, 2)
            x2 = self.head2(self.ln_f2(x)).view(x.shape[0], self.width, self.height, 2)
            x = torch.stack([x1,x2], dim=1)

            loss = 0
            if targets is not None:
                loss = F.cross_entropy(x.reshape(-1, x.size(-1)), targets.view(-1))

            return x, loss, None

        def configure_optimizers(self, cfg):
            return torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE, betas=(0.9, 0.95))

    model = ResModel(basic_model, cfg, cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)
    model.to(device)
    return model


def build_model_rt(cfg, device):
    mconf = GPTConfig(block_size=cfg.OUTPUT_WIDTH * cfg.OUTPUT_HEIGHT, num_classes=2,
                      n_layer=1, n_head=4, n_embd=128, n_recur=32,
                      all_layers=True)
    model = GPT(mconf)

    model.to(device)
    return model


if __name__ == "__main__":
    # test building the model
    cfg = setup_cfg()

    resnet_model = build_model_resnet(cfg, device=cfg.DEVICE)
    rt_model = build_model_rt(cfg, device=cfg.DEVICE)
    resnet_traj_model = build_model_traj_resnet(cfg, device=cfg.DEVICE)

    sample = torch.rand((16, 3, 350, 750), device=cfg.DEVICE)
    rt_target = torch.zeros((16, 2, 10, 10)).long().to("cuda:0")
    y = (torch.rand((16, 10, 10)) > 0.8).long().to("cuda:0")
    rt_target[:, 0, :, :] = y
    rt_target[:, 1, :, :] = -100

    t1 = rt_target[5]
    t11 = t1[0]
    t12 = t1[1]

    resnet_traj_output = resnet_traj_model(sample, targets=rt_target, sp=torch.rand((16, 10, 10)).to("cuda:0"), tp=torch.rand((16, 10, 10)).to("cuda:0"))
    resnet_output = resnet_model(sample)
    rt_output, loss, _ = rt_model(sample, rt_target)

    print()
