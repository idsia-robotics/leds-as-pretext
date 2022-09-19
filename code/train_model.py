import os
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
from model import NN
from tqdm import tqdm
from test_model import test
from datetime import datetime
from dataset import get_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, dataset_labeled, dataset_unlabeled, optimizer, batch_size, epoch, reduce_lab, model_type):
    model.train()
    losses = [[], [], []]

    size_limit = None
    batch_size_lab = 64

    lab_length = len(dataset_labeled) if reduce_lab is None else reduce_lab

    total_labeled = lab_length // batch_size_lab
    if lab_length % batch_size_lab != 0:
        total_labeled += 1

    if reduce_lab is not None:
        size_limit = total_labeled

    labeled_steps_logger = tqdm(dataset_labeled.batches(batch_size_lab, shuffle=True, size_limit=size_limit),
                                desc='  labeled step', total=total_labeled)

    if model_type != 'baseline':
        total_unlabeled = len(dataset_unlabeled) // batch_size
        if len(dataset_unlabeled) % batch_size != 0:
            total_unlabeled += 1

        unlabeled_steps_logger = tqdm(dataset_unlabeled.batches(batch_size, shuffle=True),
                                      desc='unlabeled step', total=total_unlabeled)
        for data in unlabeled_steps_logger:
            if LOG_WANDB:
                wandb.log({'epoch': epoch})

            image_off = data['image_off']
            image_on = data['image_on']

            image = torch.cat([image_off, image_on], dim=0)

            if model_type == 'upper':
                pose = data['target_proj']
                pose = torch.cat([pose, pose], dim=0)

            led = torch.cat([torch.tensor([[1.0, 0]]).repeat_interleave(image.shape[0] // 2, dim=0),
                            torch.tensor([[0, 1.0]]).repeat_interleave(image.shape[0] // 2, dim=0)],
                            dim=0).to(image.device)

            optimizer.zero_grad()

            pose_pred, led_pred = model(image)

            if unlabeled_steps_logger.n % 50 == 0:
                print("Unlabeled, prediction on last sample: ", pose_pred[-1])
                print("Unlabeled, target on last sample: ", pose[-1])

            if model_type == 'pretext':
                loss_end = torch.tensor(0, device=image.device)
                loss_pretext = torch.nn.functional.cross_entropy(led_pred, led)
            else:
                loss_end = torch.nn.functional.l1_loss(pose_pred, pose)
                loss_pretext = torch.tensor(0, device=image.device)

            loss = loss_end + loss_pretext

            loss.backward()
            optimizer.step()

            losses[0].append(loss.item())
            losses[1].append(loss_end.item())
            losses[2].append(loss_pretext.item())

    for data in labeled_steps_logger:

        if LOG_WANDB:
            wandb.log({'epoch': epoch})

        pose = data['target_proj']
        image_off = data['image_off']
        image_on = data['image_on']

        image = torch.cat([image_off, image_on], dim=0)
        pose = torch.cat([pose, pose], dim=0)
        led = torch.cat([torch.tensor([[1.0, 0]]).repeat_interleave(image.shape[0] // 2, dim=0),
                        torch.tensor([[0, 1.0]]).repeat_interleave(image.shape[0] // 2, dim=0)],
                        dim=0).to(image.device)

        optimizer.zero_grad()

        pose_pred, led_pred = model(image)
        loss_end = torch.nn.functional.l1_loss(pose_pred, pose)

        if labeled_steps_logger.n % 50 == 0:
            print("Labeled, prediction on last sample: ", pose_pred[-1])
            print("Labeled, target on last sample: ", pose[-1])

        if model_type == 'pretext':
            loss_pretext = torch.nn.functional.cross_entropy(led_pred, led)
        else:
            loss_pretext = torch.tensor(0, device=image.device)

        loss = loss_end + loss_pretext

        loss.backward()
        optimizer.step()

        losses[0].append(loss.item())
        losses[1].append(loss_end.item())
        losses[2].append(loss_pretext.item())

    return [np.mean(l) for l in losses]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_' + str(datetime.now()))
    parser.add_argument('-t', '--type', type=str, help='type of the model',
                        default='baseline')
    parser.add_argument('-bp', '--basepath', type=str, help='base path',
                        default='.')
    parser.add_argument('-f', '--filename', type=str, help='name of the dataset (.h5 file)',
                        default='ds.h5')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs of the training phase', default=100)
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=32)
    parser.add_argument('-lr', '--learning-rate', type=float,
                        help='learning rate used for the training phase', default=1e-4)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-r', '--reduce', type=int, help='number of labeled data to use',
                        default=None)
    parser.add_argument('-a', '--augment', type=str, help='whether to augment the data',
                        default='False')
    parser.add_argument('-lw', '--log-wandb', type=str, help='whether to log to wandb',
                        default='True')
    parser.add_argument('-p', '--project-name', type=str, help='name of wandb project',
                        default='leds-pretext')
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('%s = "%s"' % (k, str(v)))

    LOG_WANDB = args.log_wandb == 'True'
    AUGMENT = args.augment == 'True'

    model_id = '_all' if args.reduce is None else '_reduced' + str(args.reduce)

    base_path = args.basepath
    data_path = base_path + '/../data/processed/'
    model_path = base_path + '/../model_v3/model' + \
        model_id + f'_{args.type}_' + str(datetime.now())
    log_path = model_path + '/log'
    checkpoint_path = model_path + '/checkpoints'

    if not os.path.exists(base_path):
        raise IOError('Path "' + base_path + '" does not exist')

    os.makedirs(log_path)
    os.makedirs(checkpoint_path)

    print(f"[{str(datetime.now())}] Getting train_set...")

    # Datasets
    train_set_labeled = get_dataset(data_path + args.filename, "train_labeled", augment=AUGMENT,
                                    device=args.device)

    train_set_unlabeled = None

    if args.type != 'baseline':
        train_set_unlabeled = get_dataset(data_path + args.filename, "train_unlabeled", augment=AUGMENT,
                                          device=args.device)

    print(f"[{str(datetime.now())}] Getting val_set...")

    val_set = get_dataset(data_path + args.filename, "validation", augment=False,
                          device=args.device)

    # Model, optimizer & loss
    model = NN().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=30, factor=0.1, min_lr=1e-06, verbose=True)

    # Training
    history = pd.DataFrame()
    lr = args.learning_rate

    params = {
        "learning_rate": lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    group_name = f"{args.reduce}_labels" if args.reduce is not None else "all_labels"

    if LOG_WANDB:
        wandb.init(project=args.project_name, config=params,
                   group=group_name, name=args.type)

    epochs_logger = tqdm(range(1, args.epochs + 1), desc='epoch')

    stop_counter = 0
    min_val_loss = np.inf

    for epoch in epochs_logger:

        train_metrics = train(model, train_set_labeled, train_set_unlabeled,
                              optimizer, args.batch_size, epoch, args.reduce, args.type)

        loss, loss_end, loss_pretext = train_metrics

        val_metrics, r2, roc_auc, _ = test(model, val_set,
                                           4 * args.batch_size, verbose=True)

        val_loss, val_loss_end, val_loss_pretext = val_metrics

        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']

        history = history.append({
            'epoch': epoch,
            'loss': loss,
            'loss_end': loss_end,
            'loss_pretext': loss_pretext,
            'val_loss': val_loss,
            'val_loss_end': val_loss_end,
            'val_loss_pretext': val_loss_pretext,
            'val_r2_score': r2,
            'val_roc_auc': roc_auc,
            'lr': lr
        }, ignore_index=True)

        if LOG_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': loss,
                'train_loss_end': loss_end,
                'train_loss_pretext': loss_pretext,
                'val_loss': val_loss,
                'val_loss_end': val_loss_end,
                'val_loss_pretext': val_loss_pretext,
                'val_r2_score': r2,
                'val_roc_auc': roc_auc,
                'lr': lr,
                'stop_counter': stop_counter
            })

        log_str = 'L: %.4f VL: %.4f VLend: %.4f VLpre: %.4f' % (
            loss, val_loss, val_loss_end, val_loss_pretext
        )
        epochs_logger.set_postfix_str(log_str)

        history.to_csv(log_path + '/history.csv')

        # note: uncomment to save every checkpoint
        # checkpoint_name = '%d_%.3f_state_dict.pth' % (epoch, val_loss)
        # torch.save(model.state_dict(), checkpoint_path + '/' + checkpoint_name)

        torch.save(model.state_dict(), checkpoint_path + '/last.pth')

        if val_loss <= history['val_loss'].min():
            torch.save(model.state_dict(), checkpoint_path + '/best.pth')
