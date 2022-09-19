import os
import torch
import argparse
import numpy as np
from model import NN
from tqdm import tqdm
from dataset import get_dataset
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score


def test(model, dataset, batch_size, verbose=True, mode='validation', type='baseline'):
    losses = [[], [], []] if mode == 'validation' else [[], [], [], [], []]
    steps_logger = dataset.batches(batch_size)

    predictions_end, predictions_pretext = [], []
    targets_end, targets_pretext = [], []

    if verbose:
        steps_logger = tqdm(steps_logger,
                            desc=dataset.split + ' step',
                            total=len(dataset) // batch_size)
    model.eval()
    with torch.no_grad():
        for data in steps_logger:
            pose = data['target_proj']

            image_off = data['image_off']
            image_on = data['image_on']
            led = torch.cat([torch.tensor([[1.0, 0]]).repeat_interleave(image_off.shape[0], dim=0),
                            torch.tensor([[0, 1.0]]).repeat_interleave(image_on.shape[0], dim=0)],
                            dim=0).to(image_off.device)

            targets_pretext.extend(led.cpu().numpy())

            for image, led in zip([image_off, image_on],
                                  led):

                pose_pred, led_pred = model(image)
                targets_end.extend(pose.cpu().numpy())
                predictions_end.extend(pose_pred.detach().cpu().numpy())

                predictions_pretext.extend(led_pred.detach().cpu().numpy())

                loss_end = torch.nn.functional.l1_loss(pose_pred, pose)

                if mode == 'test':
                    loss_end_x = torch.nn.functional.l1_loss(
                        pose_pred[:, 0], pose[:, 0])
                    loss_end_y = torch.nn.functional.l1_loss(
                        pose_pred[:, 1], pose[:, 1])
                    losses[3].append(loss_end_x.item())
                    losses[4].append(loss_end_y.item())

                loss_pretext = torch.nn.functional.cross_entropy(led_pred, led)

                loss = loss_end + (type == 'pretext') * loss_pretext

                losses[0].append(loss.item())
                losses[1].append(loss_end.item())
                losses[2].append(loss_pretext.item())

                print("Val. prediction on last sample: ", pose_pred[-1])
                print("Val. target on last sample: ", pose[-1])

        targets_end = np.array(targets_end)
        predictions_end = np.array(predictions_end)

        r2_x = r2_score(targets_end[:, 0], predictions_end[:, 0])
        r2_y = r2_score(targets_end[:, 1], predictions_end[:, 1])
        r2 = (r2_x + r2_y) / 2
        pearson, _ = pearsonr(targets_end[:, 0], predictions_end[:, 0])

        if mode == 'test':
            r2 = (r2, r2_x, r2_y)

    return tuple([np.mean(l) for l in losses]), r2, roc_auc_score(targets_pretext, predictions_pretext), pearson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_abc')
    parser.add_argument('-bp', '--basepath', type=str, help='base path',
                        default='.')
    parser.add_argument('-f', '--filename', type=str, help='name of the dataset (.h5 file)',
                        default='ds.h5')
    parser.add_argument('-s', '--split', type=str, help='name of the dataset split',
                        default='test')
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=512)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-mf', '--model-folder', type=str, help="folder containing the model",
                        default='model')
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('%s = "%s"' % (k, str(v)))

    base_path = args.basepath
    data_path = base_path + '/../data/processed/'
    checkpoint_path = base_path + \
        f'/../{args.model_folder}/' + args.name + '/checkpoints'

    for path in [base_path, data_path, checkpoint_path]:
        if not os.path.exists(path):
            raise IOError('Path "' + path + '" does not exist')

    # Dataset
    dataset = get_dataset(data_path + args.filename, split=args.split,
                          augment=False, device=args.device)

    # Model
    model = NN().to(args.device)
    model.load_state_dict(torch.load(checkpoint_path + '/best.pth',
                                     map_location=args.device))

    # Testing
    metrics, r2, roc_auc = test(model, dataset,
                                args.batch_size, verbose=True, mode='test', type='baseline')

    string = 'L: %.4f Lend: %.4f Lpre: %.4f LX: %.4f LY: %.4f' % metrics

    print('%s %s |\t\t%s' % (args.name, args.split, string))
    print(f'R2 score: {round(r2[0], 3)}')
    print(f'R2 score on X: {round(r2[1], 3)}')
    print(f'R2 score on Y: {round(r2[2], 3)}')
    print(f'ROC AUC score: {round(roc_auc, 3)}')
