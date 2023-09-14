import torch
from PIL import Image
from numpy import unravel_index, stack
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits


class Autoencoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2, 4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4, 2, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(2, 1, kernel_size=1, padding=0, stride=1),
            torch.nn.ReLU(),
        )

        self.pos_head = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),
        )

        self.net = torch.nn.Sequential(self.encoder, self.decoder)
        self.loss = self.__autoencoder_loss
        self.mode = "autoencoder"

    def get_optimizer(self, lr, ratio=1e-2):
        if self.mode == "autoencoder":
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif self.mode == "position":
            self.loss = self.__pos_loss
            optimizer = torch.optim.Adam(
                [
                    {"params": self.encoder.parameters(), "lr": lr * ratio},
                    {"params": self.pos_head.parameters(), "lr": lr},
                ]
            )
        return optimizer

    def position_mode(self):
        self.net = torch.nn.Sequential(self.encoder, self.pos_head)
        self.mode = "position"

    def __ensure_input_shape(self, x):
        if len(x.shape) == 2:
            x = x.view(1, 1, *x.shape)
        elif len(x.shape) == 3:
            x = x.view(1, *x.shape)
        return x

    def predict_pos(self, x: torch.Tensor):
        x = self.__ensure_input_shape(x)
        outs: torch.Tensor = self(x)
        map_shape = outs.shape[-2:]

        outs = outs.view(outs.shape[0], -1)
        max_idx = outs.argmax(1).cpu()
        indexes = unravel_index(max_idx, map_shape)

        indexes = stack([indexes[1], indexes[0]]).T.astype("float32")
        indexes /= float(map_shape[0])
        indexes *= float(x.shape[-1])
        return indexes

    def forward(self, x):
        return self.net(x)

    def __pos_loss(self, pred, true, eps=1e-6):
        true_pooled = torch.torch.nn.functional.avg_pool2d(
            true, true.shape[-1] // pred.shape[-1]
        ).detach()
        pos_pred_sum = torch.sum(pred + eps, axis=[1, 2], keepdims=True)
        pos_pred_n = pred / pos_pred_sum
        loss = 1 - (pos_pred_n * true_pooled).sum(axis=(1, 2))
        return loss

    def __autoencoder_loss(self, true, pred):
        return torch.nn.functional.mse_loss(pred, true)


class ClipHead(torch.nn.Module):
    def __init__(self, clip_make_embeddings=None, clip_preprocess=None):
        super(ClipHead, self).__init__()

        self.dense = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.LazyLinear(out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features=400),
            torch.nn.BatchNorm1d(400),
        )

        self.map = torch.nn.Sequential(
            torch.nn.Unflatten(1, (1, 20, 20)),
            torch.nn.Sigmoid(),
        )
        self.net = torch.nn.Sequential(self.dense, self.map)

        self.clip_make_embeddings = clip_make_embeddings
        self.clip_preprocess = clip_preprocess
        self.device = next(self.parameters()).device

    def forward(self, x):
        image_array = (x * 255).to(torch.uint8).split(1)
        pil_images = [Image.fromarray(img.cpu().numpy().squeeze()) for img in image_array]
        out = torch.stack([self.clip_preprocess(img) for img in pil_images]).to(self.device)
        embeddings = self.clip_make_embeddings(out).to(torch.float32).detach()
        return self.net(embeddings)

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        self.device = next(res.parameters()).device
        return res

    def loss(self, pred, true, eps=1e-6):
        true_pooled = torch.torch.nn.functional.avg_pool2d(
            true, true.shape[-1] // pred.shape[-1]
        ).detach()
        pos_pred_sum = torch.sum(pred + eps, axis=[-1, -2], keepdims=True)
        pos_pred_n = pred / pos_pred_sum
        loss = 1 - (pos_pred_n * true_pooled).sum(axis=(-1, -2))
        return loss


class ConvBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class Frontnet(torch.nn.Module):
    def __init__(self, h=320, w=320, c=32, fc_nodes=12800):
        super(Frontnet, self).__init__()

        self.name = "Frontnet"

        self.inplanes = c
        self.width = w
        self.height = h
        self.dilation = 1
        self._norm_layer = torch.nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64

        self.conv = torch.nn.Conv2d(
            1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False
        )

        self.bn = torch.nn.BatchNorm2d(self.inplanes)
        self.relu1 = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = ConvBlock(self.inplanes, self.inplanes, stride=2)
        self.layer2 = ConvBlock(self.inplanes, self.inplanes * 2, stride=2)
        self.layer3 = ConvBlock(self.inplanes * 2, self.inplanes * 4, stride=2)

        self.dropout = torch.nn.Dropout()
        self.fc = torch.nn.Linear(fc_nodes, 4)

    def forward(self, x):
        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu1(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = l3.flatten(1)

        out = self.dropout(out)
        out = self.fc(out)
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        phi = out[:, 3]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return [x, y, z, phi]


class EDNN(torch.nn.Module):
    def __init__(self):
        super(EDNN, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1),
        )

        self.grid = torch.stack(
            torch.meshgrid([torch.arange(320)] * 2, indexing="xy"), dim=-1
        ).unsqueeze(0)

    def forward(self, x):
        return self.layers(x)[:, :1, ...]

    def loss(self, pred, pos_gt):
        raw_conf = pred
        focal = (pos_gt - torch.sigmoid(raw_conf)) ** 2
        loss = focal * bce_logits(raw_conf, pos_gt, reduction="none")
        return torch.mean(loss.sum(dim=(1, 2, 3)))

    @torch.no_grad()
    def localize(self, pos, argmax=False, threshold=None):
        pos_logits = torch.sigmoid(pos)

        pos_map_upscaled = torch.nn.functional.interpolate(
            pos_logits, scale_factor=8, mode="nearest"
        )[:, 0]

        if threshold is not None:
            threshold *= pos_map_upscaled.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
            pos_map_upscaled = torch.clamp(pos_map_upscaled - threshold, 0)

        if argmax:
            m = pos_map_upscaled.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
            pos_map_upscaled = (pos_map_upscaled == m).to(torch.float32)

        pos_map_prob = pos_map_upscaled / pos_map_upscaled.sum(dim=(-2, -1), keepdim=True)
        pos_scalar = (self.grid.to(pos_logits.device) * pos_map_prob.unsqueeze(-1)).sum(dim=(1, 2))
        return pos_scalar


class MkModel_led(torch.nn.Module):
    def __init__(self):
        super(MkModel_led, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1),
            torch.nn.Sigmoid(),
        )

        self.grid = torch.stack(
            torch.meshgrid([torch.arange(320)] * 2, indexing="xy"), dim=-1
        ).unsqueeze(0)

    def forward(self, x, return_led_map=False):
        maps = self.layers(x)
        pos, led_on = maps[:, :1], maps[:, 1:]
        led_on_prob = led_on.mean(dim=-1).mean(dim=-1)
        led_prob = torch.cat([1 - led_on_prob, led_on_prob], dim=1)

        if return_led_map:
            return led_prob, pos, led_on

        return led_prob, pos

    @torch.no_grad()
    def localize(self, pos, argmax=False, threshold=None):
        pos_map_upscaled = torch.nn.functional.interpolate(pos, scale_factor=8, mode="nearest")[
            :, 0
        ]

        if threshold is not None:
            threshold *= pos_map_upscaled.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
            pos_map_upscaled = torch.clamp(pos_map_upscaled - threshold, 0)

        # note: barycenter is assumed as default, unless argmax is True
        if argmax:
            m = pos_map_upscaled.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
            pos_map_upscaled = (pos_map_upscaled == m).to(torch.float32)

        pos_map_prob = pos_map_upscaled / pos_map_upscaled.sum(dim=(-2, -1), keepdim=True)
        pos_scalar = (self.grid.to(pos.device) * pos_map_prob.unsqueeze(-1)).sum(dim=(1, 2))
        return pos_scalar


if __name__ == "__main__":
    from torchinfo import summary

    summary(MkModel_led(), (1, 1, 320, 320))
