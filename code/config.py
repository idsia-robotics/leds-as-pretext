import torch
import argparse
from pathlib import Path
from datetime import datetime


_full_subset = ["TR1", "TE1", "TE0", "MSG", "NOV"]
_id_to_subset_mapping = dict(enumerate(_full_subset))
_subset_to_id_mapping = {v: k for k, v in _id_to_subset_mapping.items()}


def get_subset_ids(*subset):
    return [_subset_to_id_mapping[d] for d in subset]


def parse_args(*config):
    parser = argparse.ArgumentParser()

    parser.add_argument("-bp", "--basepath", type=str, help="base path", default=".")
    parser.add_argument(
        "-f", "--filename", type=str, help="filename of the dataset", default="D2D_dataset.h5"
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=str,
        nargs="+",
        help="datasets to include",
        default=[_full_subset[0]],
        choices=_full_subset,
    )
    parser.add_argument(
        "-a", "--augment", help="when set, applies data augmentations", action="store_true"
    )
    parser.add_argument("-sz", "--image-size", type=int, help="dataset image size", default=320)

    parser.add_argument(
        "-exposure",
        "--image-exposure-level",
        type=int,
        help="scaling factor for pos loss",
        default=None,
    )
    parser.add_argument(
        "-led",
        "--led",
        type=int,
        help="when set to {0, 1} loads only data with leds on or off",
        default=None,
    )

    parser.add_argument(
        "--train-sample-count", type=int, help="Number of training sample to use", default=None
    )
    parser.add_argument(
        "--train-sample-seed", type=int, help="Seed to use for sample selection", default=None
    )

    if "model" in config or "train" in config:
        parser.add_argument(
            "-n",
            "--name",
            "--model",
            type=str,
            help="name of the model",
            default="model_" + datetime.now().strftime("%Y%m%d%H%M%S"),
        )
        parser.add_argument(
            "-c", "--checkpoint", type=str, help="Checkpoint to pick the model from", default=-1
        )
        parser.add_argument(
            "-m",
            "--method",
            type=str,
            help="inference method for predicting drone location",
            choices=["amax", "baricenter"],
            default="baricenter",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            help=argparse.SUPPRESS,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )

    if "train" in config:
        parser.add_argument(
            "-e", "--epochs", type=int, help="number of epochs of the training phase", default=100
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=int,
            help="size of the batches of the training data",
            default=64,
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=float,
            help="learning rate used for the training phase",
            default=2e-3,
        )
        parser.add_argument(
            "-desc", "--description", type=str, help="Description for this model run", default=None
        )

    if "test" in config:
        parser.add_argument(
            "-c", "--checkpoint", type=str, help="Checkpoint to pick the model from", required=True
        )
        parser.add_argument("-n", "--model", type=str, help="name of the model", required=True)
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            help=argparse.SUPPRESS,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )

    if "pretext" in config:
        parser.add_argument(
            "-wpos", "--weight-pos", type=float, help="scaling factor for pos loss", default=1.0
        )
        parser.add_argument(
            "-fpos",
            "--fraction-pos",
            type=float,
            help="fraction of training-set with know position",
            default=1.0,
        )
        parser.add_argument(
            "-fled",
            "--fraction-led",
            type=float,
            help="fraction of training-set with know led status",
            default=1.0,
        )

    if "autoencoder" in config:
        parser.add_argument(
            "--position-mode",
            action="store_true",
            help="If true, will load the model in position inference mode. \nThe output of this network will be a position map.",
        )
        parser.add_argument(
            "--pre-trained-name", type=str, help="Name for the pre-trained model", default=None
        )
        parser.add_argument("--bottleneck-size", type=int, default=512)

    if "clip" in config:
        parser.add_argument("--clip-base-folder", type=str, help="Base folder for the CLIP model")

    if "ednn" in config:
        parser.add_argument(
            "--pre-trained-name", type=str, help="Name for the pre-trained model", default=None
        )

    args = parser.parse_args()

    args.basepath = Path(args.basepath)

    return args
