import clip
import torch
import config
from model import ClipHead
from statistics import mean
from dataset import get_dataset
from test_led_pretext import test
from torch.utils.tensorboard import SummaryWriter


def train(model: ClipHead, dataloader, optimizer, device):
    losses = []
    model.train()

    loss_fn = None
    batch_target_key = None
    batch_input_key = "image"

    loss_fn = model.loss
    batch_target_key = "pos_map"

    for _, batch in enumerate(dataloader):
        x = batch[batch_input_key].to(device)
        y = batch[batch_target_key].to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(pred=outputs, true=y)
        loss.mean().backward()
        optimizer.step()
        losses.append(loss.detach().mean().item())
    return mean(losses)


if __name__ == "__main__":
    import tqdm
    from pathlib import Path

    args = config.parse_args("train", "clip")
    args.basepath = Path(args.basepath)

    clip_preprocessor = None

    # Dataset

    train_dataset = get_dataset(
        args.basepath / "data" / args.filename,
        args.subset,
        augment=True,
        size=args.image_size,
        exposure=args.image_exposure_level,
        samples_count=args.train_sample_count,
        samples_count_seed=args.train_sample_seed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    val_dataset = get_dataset(args.basepath / "data" / args.filename, subset=["TE1"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )

    # Model & Optimizer

    clip_model, preprocess = clip.load(
        "ViT-B/32", device=args.device, download_root=args.basepath / args.clip_base_folder
    )
    clip_model.eval()
    clip_model = clip_model.to(args.device)
    clip_preprocessor = preprocess
    model = ClipHead(clip_model.encode_image, preprocess).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(args.epochs * 0.66),
            int(args.epochs * 0.75),
            int(args.epochs * 0.9),
        ],
        gamma=0.5,
    )

    # Summary

    checkpoint_path: Path = args.basepath / "models" / (args.name + "_" + model.__class__.__name__)
    checkpoint_path.mkdir(exist_ok=False)

    log_path = checkpoint_path / "log"
    log_path.mkdir(exist_ok=False)
    writer = SummaryWriter(log_dir=log_path)
    print(f"Training model {model.__class__.__name__} with name {checkpoint_path.parts[-1]}")
    print()

    # Training
    logger = tqdm.trange(args.epochs)

    loss = 0

    for epoch in logger:
        loss = train(
            model,
            train_dataloader,
            optimizer,
            args.device,
        )

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("LR/train", scheduler.get_last_lr()[-1], epoch)
        logger.set_postfix_str(f"L {loss:.4f}")
        scheduler.step()

        checkpoint_file = checkpoint_path / f"{epoch}.tar"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "model_name": model.__class__.__name__,
                "model_comment": args.description,
            },
            f=checkpoint_file,
        )
        print()

        # Validation
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

    checkpoint_file = checkpoint_path / "model.tar"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "model_name": model.__class__.__name__,
            "model_comment": args.description,
        },
        f=checkpoint_file,
    )
