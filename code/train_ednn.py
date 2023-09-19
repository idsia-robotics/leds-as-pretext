import torch
from model import EDNN
from dataset import get_dataset
from test_led_pretext import test
from statistics import mean, median
from torch.utils.tensorboard import SummaryWriter


def train(model: EDNN, dataloader, optimizer, device):
    losses = dict(loss=[], dist_bari=[], dist_amax=[])
    model.train()

    for batch in dataloader:
        x = batch["image"].to(device)
        uv = torch.stack([batch["proj_u"], batch["proj_v"]], dim=1).to(device)
        pos = batch["y"][:, 0].to(device)

        optimizer.zero_grad()

        preds = model(x)

        pos_gt_pooled = torch.torch.nn.functional.avg_pool2d(pos, 8).unsqueeze(1)
        loss = model.loss(pos_gt=pos_gt_pooled, pred=preds)

        pos_scalar_bari = model.localize(preds, argmax=False)
        pos_scalar_amax = model.localize(preds, argmax=True)

        distance_bari = torch.norm(pos_scalar_bari - uv, dim=-1)
        distance_amax = torch.norm(pos_scalar_amax - uv, dim=-1)

        loss.backward()
        optimizer.step()

        losses["loss"].append(loss.item())
        losses["dist_bari"] += distance_bari.cpu().tolist()
        losses["dist_amax"] += distance_amax.cpu().tolist()

    return {
        "loss": mean(losses["loss"]),
        "dist_bari": median(losses["dist_bari"]),
        "dist_amax": median(losses["dist_amax"]),
        "dist_bari_hist": torch.tensor(losses["dist_bari"]),
        "dist_amax_hist": torch.tensor(losses["dist_amax"]),
    }


if __name__ == "__main__":
    import tqdm
    from config import parse_args

    args = parse_args("train", "pretext", "ednn")

    if args.batch_size != 5:
        print(f"A batch size of {args.batch_size} was given.")
        print("A value of 5 is recomended in order to train on the *synthetic* dataset.")
        continue_choice = ""
        while continue_choice.strip().lower() not in ["y", "n"]:
            continue_choice = input("Continue anyways? (Y/N)")
            if continue_choice.strip().lower() == "n":
                print("Exiting...")
                exit()

    checkpoint_path = args.basepath / "models" / args.name
    checkpoint_path.mkdir(exist_ok=False)

    log_path = checkpoint_path / "log"
    log_path.mkdir(exist_ok=False)

    # Dataset

    train_dataset = get_dataset(
        args.basepath / "data" / args.filename,
        subset=["TR1"],
        samples_count_seed=args.train_sample_seed,
        samples_count=args.train_sample_count,
        augment=args.augment,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    # Model & Optimizer

    model = EDNN()

    # We only care to load the validation set on the fine tuning phase.
    validation_on = args.pre_trained_name is not None
    if validation_on:
        print(f"Loading pre trained with name: {args.pre_trained_name}")
        checkpoint = torch.load(args.basepath / "models" / args.pre_trained_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_dataset = get_dataset(args.basepath / "data" / "D2D_dataset.h5", subset=["TE1"])
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
        )
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def lr_scheduler(epoch):
        if epoch < 2:  # warm-up epochs
            return 5e-5 * (10**epoch)
        else:
            return args.learning_rate * (1 - ((epoch - 2) / (args.epochs - 2)) ** 2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_scheduler,
    )

    # todo: uncomment to load checkpoint
    # checkpoint = torch.load(args.basepath / "models" / "led01" / "60.tar")
    # model.load_state_dict(checkpoint["model_state_dict"])

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
        )
        if validation_on:
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                val_metrics = test(
                    model,
                    val_dataloader,
                    args.device,
                    w_pos=args.weight_pos,
                    frac_pos=args.fraction_pos,
                    frac_led=args.fraction_led,
                )
                writer.add_scalar("validation/dist_bari", val_metrics["dist_bari"], epoch)
                writer.add_scalar("validation/loss_pos", val_metrics["loss_pos"], epoch)
                writer.add_scalar("validation/dist_amax", val_metrics["dist_amax"], epoch)
                writer.add_scalar(
                    "validation/precision_bari_30", val_metrics["precision_bari_30"], epoch
                )
                writer.add_scalar(
                    "validation/precision_amax_30", val_metrics["precision_amax_30"], epoch
                )
                writer.add_histogram(
                    "validation/dist_bari_hist", val_metrics["dist_bari_hist"], epoch, bins=320
                )
                writer.add_histogram(
                    "validation/dist_amax_hist", val_metrics["dist_amax_hist"], epoch, bins=320
                )

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("train/loss", metrics["loss"], epoch)
        writer.add_scalar("train/dist_bari", metrics["dist_bari"], epoch)
        writer.add_scalar("train/dist_amax", metrics["dist_amax"], epoch)
        writer.add_histogram("train/dist_bari_hist", metrics["dist_bari_hist"], epoch, bins=320)
        writer.add_histogram("train/dist_amax_hist", metrics["dist_amax_hist"], epoch, bins=320)
        torch.save(
            {
                "epoch": epoch,
                "model_name": model.__class__.__name__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f=checkpoint_path / f"{epoch}.tar",
        )
