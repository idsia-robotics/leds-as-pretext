import torch
from model import MkModel_led
from dataset import get_dataset
from statistics import mean, median
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def test(model, dataloader, device, w_pos, frac_pos, frac_led):
    losses = dict(
        loss_pre=[],
        loss_pos=[],
        auc=[],
        kp_auc=[],
        dist_bari=[],
        dist_amax=[],
        precision_bari_30=[],
        precision_amax_30=[],
    )

    model.eval()
    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["led_status"].flatten().to(torch.int64).to(device)
        uv = torch.stack([batch["proj_u"], batch["proj_v"]], dim=1).to(device)
        pos_gt = batch["y"][:, 0].to(device)
        pos_gt = torch.torch.nn.functional.avg_pool2d(pos_gt, 8)

        led_prob, pos_map, led_map = model(x, return_led_map=True)

        pos_scalar_bari = model.localize(pos_map, argmax=False)
        pos_scalar_amax = model.localize(pos_map, argmax=True)

        loss_led = torch.nn.functional.cross_entropy(input=led_prob, target=y)
        loss_pos = torch.nn.functional.mse_loss(pos_gt.unsqueeze(1), pos_map)
        loss_pos = loss_pos / frac_pos

        auc = roc_auc_score(y.cpu().numpy(), led_prob[:, 1].cpu().numpy())
        distance_bari = torch.norm(pos_scalar_bari - uv, dim=-1)
        distance_amax = torch.norm(pos_scalar_amax - uv, dim=-1)
        precision_bari_30px = (distance_bari < 30).to(torch.float32)
        precision_amax_30px = (distance_amax < 30).to(torch.float32)

        # note: known pos led auc
        drone_and_led_on = led_map * pos_gt
        kp_led_prob = drone_and_led_on.mean(dim=(-2, -1)) / (1e-6 + pos_gt.mean(dim=(-2, -1)))
        kp_auc = roc_auc_score(y.cpu().numpy(), kp_led_prob[:, 0].cpu().numpy())

        losses["loss_pre"].append(loss_led.item())
        losses["loss_pos"].append(loss_pos.item())
        losses["auc"].append(auc)
        losses["kp_auc"].append(kp_auc)
        losses["dist_bari"] += distance_bari.cpu().tolist()
        losses["dist_amax"] += distance_amax.cpu().tolist()
        losses["precision_bari_30"] += precision_bari_30px.cpu().tolist()
        losses["precision_amax_30"] += precision_amax_30px.cpu().tolist()

    return {
        "loss_pre": mean(losses["loss_pre"]),
        "loss_pos": mean(losses["loss_pos"]),
        "auc": mean(losses["auc"]),
        "kp_auc": mean(losses["kp_auc"]),
        "dist_bari": median(losses["dist_bari"]),
        "dist_amax": median(losses["dist_amax"]),
        "dist_bari_hist": torch.tensor(losses["dist_bari"]),
        "dist_amax_hist": torch.tensor(losses["dist_amax"]),
        "precision_bari_30": mean(losses["precision_bari_30"]),
        "precision_amax_30": mean(losses["precision_amax_30"]),
    }


if __name__ == "__main__":
    from config import parse_args

    args = parse_args("model")

    checkpoint = args.basepath / "models" / args.name / f"{args.checkpoint}.tar"

    # Dataset

    dataset = get_dataset(
        args.basepath / "data" / args.filename,
        args.subset,
        augment=args.augment,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )

    # Model

    model = MkModel_led()
    checkpoint = torch.load(checkpoint)

    model.to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Testing

    metrics = test(model, dataloader, args.device, w_pos=1, frac_pos=1, frac_led=1)

    # todo: handle list retruned from test()
    for k, v in metrics.items():
        print(f"{k.upper().replace('_', ' ')} {v:.4f}", end=", ")
