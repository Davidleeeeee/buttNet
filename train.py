import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.data_loading_butterfly import BasicDatasetBF, CarvanaDatasetBF
from butterfly_net import ButterflyNet


class SquareLoss(nn.Module):
    def __init__(self, ):
        super(SquareLoss, self).__init__()
        return

    def forward(self, x, y):
        loss = torch.sqrt(y) - torch.sqrt(x)
        return loss


def train_net(net,
              device,
              dir_mask=Path(r'E:\文档\picture_crop2'),
              dir_img=Path(r'E:\文档\picture_crop2'),
              dir_checkpoint=Path('./checkpoints/'),
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDatasetBF(dir_img, dir_mask)
    except (AssertionError, RuntimeError):
        dataset = BasicDatasetBF(dir_img, dir_mask)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-9, momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.L1Loss()
    global_step = 0

    # 5. Begin training
    for index, epoch in enumerate(range(epochs)):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                origen = batch['image_origen']
                true_masks = batch['mask']
                origen = origen.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred1, masks_pred2 = net(origen)
                    loss = criterion(masks_pred1, true_masks) + criterion(masks_pred2, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward(loss.clone().detach())
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(origen.shape[0])
                global_step += 1
                epoch_loss += torch.sum(loss)
                pbar.set_postfix(**{'loss (batch)': epoch_loss})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'ok4_7Patches_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the buttNet on images and target masks')
    parser.add_argument('--images', type=str, default='./data',
                        help='train-set')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = ButterflyNet(n_classes=3, layer_numbers=4)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net = net.to(device=device)
    try:
        train_net(net=net,
                  dir_mask=args.images,
                  dir_img=args.images,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
