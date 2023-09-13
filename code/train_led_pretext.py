import torch
from model import MkModel_led
from dataset import get_dataset
from test_led_pretext import test
from statistics import mean, median
from torch.utils.tensorboard import SummaryWriter


def train(model, dataloader, optimizer, device, w_pos, frac_pos, frac_led):
    losses = dict(loss=[], loss_pos=[], loss_pre=[], loss_spar=[], dist_bari=[], dist_amax=[])
    model.train()

    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["led_status"].flatten().to(torch.int64).to(device)
        uv = torch.stack([batch["proj_u"], batch["proj_v"]], dim=1).to(device)
        pos = batch["y"][:, 0].to(device)

        labeled_mask = batch["pretext_labeled"].to(device)
        # labeled_mask = torch.ones_like(labeled_mask).to(bool)  # todo: uncomment for upperbound

        optimizer.zero_grad()

        led_prob, pos_map = model(x)

        pos_scalar_bari = model.localize(pos_map, argmax=False)
        pos_scalar_amax = model.localize(pos_map, argmax=True)

        loss_pos = 0
        loss_led = torch.nn.functional.cross_entropy(input=led_prob, target=y)
        loss_led = loss_led / frac_led
        loss_sparsity = torch.tensor(0.0, device=device)

        pos_pooled = torch.torch.nn.functional.avg_pool2d(pos, 8)
        pos_pooled = pos_pooled.unsqueeze(1)
        loss_pos = torch.nn.functional.mse_loss(pos_pooled, pos_map, reduction="none")
        loss_pos = loss_pos.mean(dim=(-2, -1))[:, 0]
        loss_pos = torch.mean(loss_pos * labeled_mask.to(torch.float32))
        loss_pos = loss_pos / frac_pos

        loss = (1 - w_pos) * loss_led + w_pos * loss_pos

        distance_bari = torch.norm(pos_scalar_bari - uv, dim=-1)
        distance_amax = torch.norm(pos_scalar_amax - uv, dim=-1)

        loss.backward()
        optimizer.step()

        losses["loss"].append(loss.item())
        losses["loss_pre"].append(loss_led.item())
        losses["loss_pos"].append(loss_pos.item())
        losses["loss_spar"].append(loss_sparsity.item())
        losses["dist_bari"] += distance_bari.cpu().tolist()
        losses["dist_amax"] += distance_amax.cpu().tolist()

    return {
        "loss": mean(losses["loss"]),
        "loss_pre": mean(losses["loss_pre"]),
        "loss_pos": mean(losses["loss_pos"]),
        "loss_spar": mean(losses["loss_spar"]),
        "dist_bari": median(losses["dist_bari"]),
        "dist_amax": median(losses["dist_amax"]),
        "dist_bari_hist": torch.tensor(losses["dist_bari"]),
        "dist_amax_hist": torch.tensor(losses["dist_amax"]),
    }


if __name__ == "__main__":
    import tqdm
    from config import parse_args

    args = parse_args("train", "pretext")

    checkpoint_path = args.basepath / "models" / args.name
    checkpoint_path.mkdir(exist_ok=False)

    log_path = checkpoint_path / "log"
    log_path.mkdir(exist_ok=False)

    # Dataset

    train_dataset = get_dataset(args.basepath / "data" / args.filename, args.subset, augment=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    val_dataset = get_dataset(args.basepath / "data" / args.filename, ["TE1"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    # Model & Optimizer

    model = MkModel_led()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: {
            0: 1e-2,
            1: 2e-3,
            2: 4e-4,
            3: 8e-5,
            4: 8e-5,
        }[e // (args.epochs // 4)],
    )

    # Training

    writer = SummaryWriter(log_dir=log_path, flush_secs=30)

    writer.add_scalar("train/weight/pos", args.weight_pos)
    writer.add_scalar("train/weight/frac_pos", args.fraction_pos)
    writer.add_scalar("train/weight/frac_led", args.fraction_led)

    torch.save(
        {
            "epoch": -1,
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f=checkpoint_path / "-1.tar",
    )

    for epoch in tqdm.trange(args.epochs):
        metrics = train(
            model,
            train_dataloader,
            optimizer,
            args.device,
            w_pos=args.weight_pos,
            frac_pos=args.fraction_pos,
            frac_led=args.fraction_led,
        )
        val_metrics = test(
            model,
            val_dataloader,
            args.device,
            w_pos=args.weight_pos,
            frac_pos=args.fraction_pos,
            frac_led=args.fraction_led,
        )
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("train/loss", metrics["loss"], epoch)
        writer.add_scalar("train/loss_pos", metrics["loss_pos"], epoch)
        writer.add_scalar("train/loss_pre", metrics["loss_pre"], epoch)
        writer.add_scalar("train/dist_bari", metrics["dist_bari"], epoch)
        writer.add_scalar("train/dist_amax", metrics["dist_amax"], epoch)
        writer.add_scalar("validation/auc_led", val_metrics["auc"], epoch)
        writer.add_scalar("validation/kp_auc_led", val_metrics["kp_auc"], epoch)
        writer.add_scalar("validation/loss_pre", val_metrics["loss_pre"], epoch)
        writer.add_scalar("validation/loss_pos", val_metrics["loss_pos"], epoch)
        writer.add_scalar("validation/dist_bari", val_metrics["dist_bari"], epoch)
        writer.add_scalar("validation/dist_amax", val_metrics["dist_amax"], epoch)
        writer.add_scalar("validation/precision_bari_30", val_metrics["precision_bari_30"], epoch)
        writer.add_scalar("validation/precision_amax_30", val_metrics["precision_amax_30"], epoch)
        writer.add_histogram("train/dist_bari_hist", metrics["dist_bari_hist"], epoch, bins=320)
        writer.add_histogram("train/dist_amax_hist", metrics["dist_amax_hist"], epoch, bins=320)
        writer.add_histogram(
            "validation/dist_bari_hist", val_metrics["dist_bari_hist"], epoch, bins=320
        )
        writer.add_histogram(
            "validation/dist_amax_hist", val_metrics["dist_amax_hist"], epoch, bins=320
        )

        torch.save(
            {
                "epoch": epoch,
                "model_name": model.__class__.__name__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f=checkpoint_path / f"{epoch}.tar",
        )
