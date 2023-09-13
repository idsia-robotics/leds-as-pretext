import torch


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
