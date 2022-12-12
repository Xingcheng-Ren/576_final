from src.archs import *
from src.errfuncs import *
from src.dset import *

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
tr = torch


def train_model(model, dataloaders, criterion, optimizer, opath, num_epochs=35):
    val_loss_history = []
    train_loss_history = []

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # experiment.set_epoch(epoch)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        for phase in phases:
            running_loss = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode -> activate droput layers and batch norm
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                # print('inputs.shape:', inputs.shape, 'targets.shape:', targets.shape)
                # print(targets)
                for count, item in enumerate(inputs):
                    inputs[count] = item.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
            else:
                train_loss_history.append(epoch_loss)

        torch.save(model.state_dict(), f'/home/kuangjie/code/kuangjie-framework/ubfc_checkpoints/model_ep{epoch}.pt')
        print()


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='DeepPhys, PhysNet, RateProbEst')
    parser.add_argument('--loss', type=str, help='L1, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--intervals', type=int, nargs='+', help='indices: train_start, train_end, val_start, val_end, shift_idx')
    parser.add_argument('--logger_name', type=str, help='project name for commet ml experiment')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoints will be saved in this directory")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--img_size', type=int, default=128, help='size of image')
    parser.add_argument('--time_depth', type=int, default=128, help='time depth for PhysNet')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--img_augm', type=bool, default=False, help='image augmentation (flip, color jitter)')
    parser.add_argument('--freq_augm', type=bool, default=False, help='apply frequency augmentation')

    args = parser.parse_args()

    # create output dir
    if args.checkpoint_dir:
        try:
            os.makedirs(f'checkpoints/{args.checkpoint_dir}')
            print("Output directory is created")
        except FileExistsError:
            reply = input('Override existing weights? [y/n]')
            if reply == 'n':
                print('Add another outout path then!')
                exit(0)

    # Add the following code anywhere in your machine learning file
    # experiment = Experiment(api_key="", project_name=args.logger_name, workspace="")

    hyper_params = {
        "model": args.model,
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_dir": args.checkpoint_dir,
        "loss_fn": args.loss,
        "time_depth": args.time_depth,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "n_workers": args.n_cpu,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "database": args.data,
        "intervals": args.intervals,
        "img_augm": args.img_augm,
        "freq_augm": args.freq_augm
    }

    # experiment.log_parameters(hyper_params)

    # Fix random seed for reproducability
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --------------------------------------
    # Dataset and dataloader construction
    # --------------------------------------
    loader_device = None    # if multiple workers yolo works only on cpu
    if args.n_cpu == 0:
        loader_device = torch.device('cuda')
    else:
        loader_device = torch.device('cpu')

    testset = trainset = None
    if args.model == 'PhysNet':
        print('Constructing data loader for PhysNet architecture...')
        # chose label type for specific loss function
        if args.loss == 'SNR' or args.loss == 'Laplace' or args.loss == 'Gauss':
            ref_type = 'PulseNumerical'
            print('\nPulseNumerical reference type chosen!')
        else:
            ref_type = 'PPGSignal'
            print('\nPPGSignal reference type chosen!')
        
        if args.data == 'UBFC':

            trainset = Dataset4DFromUBFC(labels=ref_type,
                                        device=loader_device,
                                        start=args.intervals[0], end=args.intervals[1],
                                        augment=args.img_augm,
                                        augment_freq=args.freq_augm)

            testset = Dataset4DFromUBFC(labels=ref_type,
                                        device=loader_device,
                                        start=args.intervals[2], end=args.intervals[3],
                                        augment=False,
                                        augment_freq=False)
    else:
        print('Error! No such model.')
        exit(666)

    # Construct DataLoaders
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_cpu,
                             pin_memory=True)

    testloader = DataLoader(testset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_cpu,
                            pin_memory=True)

    dataloaders = {'train': trainloader, 'val': testloader}
    print('\nDataLoaders succesfully constructed!')

    # --------------------------
    # Load model
    # --------------------------
    if args.model == 'DeepPhys':
        model = DeepPhys()
    elif args.model == 'PhysNet':
        model = PhysNetED()
    else:
        print('\nError! No such model. Choose from: DeepPhys, PhysNet')
        exit(666)

    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = tr.nn.DataParallel(model)
    
    # If there are pretrained weights, initialize model
    if args.pretrained_weights:
        model.load_state_dict(tr.load(args.pretrained_weights))
        print('\nPre-trained weights are loaded for PhysNet!')

    # Copy model to working device
    model = model.to(device)

    # --------------------------
    # Define loss function
    # ---------------------------
    # 'L1, MSE, NegPea, SNR, Gauss, Laplace'
    loss_fn = None
    if args.loss == 'L1':
        loss_fn = nn.L1Loss()
    elif args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss == 'NegPea':
        loss_fn = NegPeaLoss()
    elif args.loss == 'SNR':
        loss_fn = SNRLoss()
    elif args.loss == 'Gauss':
        loss_fn = GaussLoss()
    elif args.loss == 'Laplace':
        loss_fn = LaplaceLoss()
    else:
        print('\nError! No such loss function. Choose from: L1, MSE, NegPea, SNR, Gauss, Laplace')
        exit(666)

    # ----------------------------
    # Initialize optimizer
    # ----------------------------
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    # -----------------------------
    # Start training
    # -----------------------------
    train_model(model, dataloaders, criterion=loss_fn, optimizer=opt, opath=args.checkpoint_dir, num_epochs=args.epochs)

    # experiment.end()

    print('\nTraining is finished without flaw!')
