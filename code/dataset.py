import math
import h5py
import torch
import torchvision
import numpy as np
from config import get_subset_ids
from pathlib import Path, PosixPath
from torchvision.transforms import functional as F, InterpolationMode


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, keys=None, transform=lambda x: x, libver="latest"):
        filename = Path(filename)

        if not filename.is_file():
            raise FileNotFoundError(f'Dataset "{filename}" does not exist')

        self.h5f = h5py.File(filename, "r", libver=libver)
        self.data = self.h5f

        if keys is None:
            keys = list(map(str, self.data.keys()))
        else:
            keys = [k for k in keys if k in self.data.keys()]

        if len(keys) == 0:
            raise ValueError(f'Dataset "{filename}" does not contain any of expected column names')

        self.keys = keys
        self.transform = transform
        self.length = len(self.data[self.keys[0]])

        length = 1000  # todo: change length of labeled training set
        self.pretext_labeled_indices = (
            33942 + np.random.default_rng(seed=0).permutation(27700)[:length]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, slice):
        batch = self.transform({k: self.data[k][slice].astype(float) for k in self.keys})
        batch["pretext_labeled"] = np.isin(slice, self.pretext_labeled_indices)
        return batch

    def __del__(self):
        if hasattr(self, "h5f"):
            self.h5f.close()


class PrepareForTrainingTransform:
    def __init__(self, size: int, dtype=torch.float32):
        self.size = size
        self.dtype = dtype
        self.X, self.Y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
        self.grid = torch.stack([self.X, self.Y], dim=0)
        self.nov_id = get_subset_ids("NOV")[0]

    def get_pos(self, p, decay_factor=0.2):
        is_batched = p.dim() == 2
        grid = self.grid[None] if is_batched else self.grid
        distance = torch.pow(grid - p[..., None, None], 2)
        distance = torch.sqrt(distance.sum(dim=1 * is_batched, keepdim=True))
        return (1 - torch.tanh((distance - 5) * decay_factor)) / 2

    def get_pos_punctual(self, p):
        pos_up_left_pixel = p - (p % 8)
        pos_up_left_pixel = torch.round(pos_up_left_pixel).type(torch.int16)
        is_batched = p.dim() == 2

        if is_batched:
            results = torch.zeros((p.shape[0], self.size, self.size))
            y_pos = tuple(pos_up_left_pixel[:, 1])
            x_pos = tuple(pos_up_left_pixel[:, 0])
            sample = tuple(torch.arange(p.shape[0]))
            results[sample, y_pos : y_pos + 8, x_pos : x_pos + 8] = 1
        else:
            results = torch.zeros((1, self.size, self.size))
            y = pos_up_left_pixel[1]
            x = pos_up_left_pixel[0]
            results[0, y : y + 8, x : x + 8] = 1

        return results + 1e-10

    def get_reshaped(self, x, shape):
        # note: assumes x.dim() <= len(shape)
        x = x.view(*x.shape, *(1 for _ in range(len(shape) - x.dim())))
        return x.expand(shape)

    def __call__(self, batch):
        batch = {k: torch.tensor(v, dtype=self.dtype).contiguous() for k, v in batch.items()}

        batch["image"] /= 255.0

        if self.size != batch["image"].shape[-1]:
            raise ValueError(
                "Cannot load images with a different resolution than the stored data's one."
            )

        pos = self.get_pos(torch.stack([batch["proj_u"], batch["proj_v"]], dim=-1))

        batch["y"] = torch.cat(
            [
                pos,
                self.get_reshaped(batch["led_status"], pos.shape),
                self.get_reshaped(batch["pose_rel"][..., 0], pos.shape),
            ],
            dim=0,
        )

        batch["distance"] = batch["pose_rel"][..., 0]
        batch["pos_map"] = pos
        nov = np.array(np.where(batch["subset"] == self.nov_id))
        batch["led_status"][nov] = -1

        return batch


class SimplexNoiseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, batch):
        return batch


class RandomRotTranslTransform:
    def __init__(self, max_angle, max_translate):
        self.max_angle = max_angle
        self.max_translate = max_translate

    def __call__(self, batch):
        y = batch["y"]
        image = batch["image"]
        size = image.size(-1)

        angle = (2 * torch.rand(1, device=y.device) - 1) * self.max_angle
        translate = (2 * torch.rand(2, device=y.device) - 1) * self.max_translate * size

        sin = torch.sin(angle * math.pi / 180).item()
        cos = torch.cos(angle * math.pi / 180).item()
        u = batch["proj_u"] - size / 2
        v = batch["proj_v"] - size / 2

        batch["proj_u"] = u * cos - v * sin + translate[0].item() + size / 2
        batch["proj_v"] = u * sin + v * cos + translate[1].item() + size / 2

        angle = angle.float().item()
        translate = translate.tolist()

        y = F.affine(
            y, angle, translate, scale=1, shear=(0, 0), interpolation=InterpolationMode.BILINEAR
        )
        image = F.affine(
            image, angle, translate, scale=1, shear=(0, 0), interpolation=InterpolationMode.BILINEAR
        )

        batch["y"] = y
        batch["image"] = image

        return batch


class RandomHorizontalFlip:
    def __init__(self, size):
        self.size = size

    def __call__(self, batch):
        if torch.rand(1) < 0.5:
            batch["proj_u"] = self.size - batch["proj_u"]
            batch["y"] = F.hflip(batch["y"])
            batch["image"] = F.hflip(batch["image"])

        return batch


def get_dataset(
    filename: PosixPath,
    subset,
    size=320,
    augment=False,
    led=None,
    exposure=None,
    samples_count=None,
    samples_count_seed=None,
):
    kwargs = {
        "keys": ["pose_rel", "led_status", "proj_u", "proj_v", "image", "subset"],
        "libver": "v112",
        "transform": [PrepareForTrainingTransform(size)],
    }

    if augment:
        kwargs["transform"].append(RandomHorizontalFlip(size))
        kwargs["transform"].append(RandomRotTranslTransform(max_angle=9, max_translate=0.1))
        kwargs["transform"].append(SimplexNoiseTransform(size))

    kwargs["transform"] = torchvision.transforms.Compose(kwargs["transform"])

    dataset = HDF5Dataset(filename, **kwargs)
    mask = torch.isin(
        torch.from_numpy(dataset.data["subset"][...]), torch.tensor(get_subset_ids(*subset))
    )

    if led is not None:
        print(mask.sum())
        mask = mask & torch.from_numpy(dataset.data["led_status"][...] == led)[:, 0]
        print(mask.sum())

    if exposure is not None:
        mask = mask & torch.from_numpy(dataset.data["exposure"][...] == exposure)[:, 0]

    if samples_count is not None:
        if samples_count == 0:
            mask = np.zeros_like(mask)
        else:
            seed = samples_count_seed
            rng = np.random.default_rng(seed)
            selected_so_far_idx = torch.where(mask)[0]
            picked = rng.choice(selected_so_far_idx, samples_count, replace=False)
            select_mask = torch.zeros_like(mask)
            select_mask[picked.tolist()] = True
            mask = mask & select_mask

    return torch.utils.data.Subset(dataset, torch.arange(len(dataset))[mask])


if __name__ == "__main__":
    from pprint import pprint
    from config import parse_args

    args = parse_args()

    dataset = get_dataset(args.basepath / "data" / args.filename, args.subset, augment=True)

    def describe(v):
        return f"{v.shape} {v.dtype}"

    print(f"{args.filename} length {len(dataset)}")
    pprint({k: describe(v) for k, v in dataset[0].items()})
