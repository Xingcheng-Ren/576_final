from src.archs import *
from src.dset import *
from src.errfuncs import *

import argparse
import numpy as np
import torch as tr
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

def fft_calculate(signal):
    minFreq = 0.5
    maxFreq = 8
    framerate = 20.0
    signal = signal - np.mean(signal)
    fft_data = np.fft.rfft(signal, 2048) # FFT
    fft_data = np.abs(fft_data)
    freq = np.fft.rfftfreq(2048, 1.0/framerate) # Frequency data
    inds= np.where((freq < minFreq) | (freq > maxFreq))[0]
    fft_data_another = fft_data.copy()
    fft_data[inds] = 0
    max_index = np.argmax(fft_data)
    # hr = framerate / 2048 * max_index * 60
    hr = freq[max_index] * 60
    return fft_data_another, max_index, hr

def eval_model(model, testloader, criterion, oname):
    result     = []
    total_loss = []
    hr_outputs = []
    hr_targets = []
    for batch_id, (inputs, targets) in enumerate(tqdm(testloader)):

        # Copy data to device
        for count, item in enumerate(inputs):
            inputs[count] = item.to(device)
        targets = targets.to(device)

        with tr.no_grad():
            outputs = model(inputs).squeeze()
            # print(f'outputs.shape: {outputs.shape}')

            if criterion is not None:
                print('outputs:', outputs.shape)
                print('targets:', targets.shape)
                loss = criterion(outputs, targets)
                print(f'Current loss: {loss.item()}')

            # print(outputs.shape)
            test_outputs = outputs.cuda().data.cpu().numpy()
            test_targets = targets.cuda().data.cpu().numpy()
            test_inputs  = inputs.cuda().data.cpu().numpy()
            for k in range(test_inputs.shape[0]):
                spectrum_output, maxid_output, hr_output = fft_calculate(test_outputs[k, :])
                spectrum_target, maxid_target, hr_target = fft_calculate(test_targets[k, :])
                hr_outputs.append(hr_output)
                hr_targets.append(hr_target)

                fig, axes = pyplot.subplots(2, 2, figsize=(18, 18))
                axes[0,0].plot(test_outputs[k,:])
                axes[0,1].plot(spectrum_output)
                axes[1,0].plot(test_targets[k,:])
                axes[1,1].plot(spectrum_target)
                axes[0,0].set_title(str(hr_output))
                axes[0,1].set_title(str(maxid_output))
                axes[1,0].set_title(str(hr_target))
                axes[1,1].set_title(str(maxid_target))
                pyplot.savefig('physnet-ubfc/'+ str(batch_id) + '-' + str(k) + ".png")

                # save network output
                result.extend(outputs.data.cpu().numpy().flatten().tolist())

        if criterion is not None:
            total_loss.append(loss.item())

    if criterion is not None:
        total_loss = np.nanmean(total_loss)
        print(f'\n------------------------\nTotal loss: {total_loss}\n-----------------------------')

    np.savetxt(f'outputs/{oname}', np.array(result))
    
    df = pd.DataFrame({'groundtruth': hr_targets, 'heartrate': hr_outputs})
    df.to_excel('/home/kuangjie/code/kuangjie-framework/physnet_ubfc.xlsx', sheet_name='UBFC', index=False)
    print('Result saved!')


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='DeepPhys, PhysNet')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--interval', type=int, nargs='+',
                        help='indices: val_start, val_end, shift_idx; if not given -> whole dataset')
    parser.add_argument("--weight", type=str, help="model weight path")
    parser.add_argument('--loss', type=str, default=None, help='Loss function: L1, RMSE, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--ofile_name", type=str, help="output file name")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')

    args = parser.parse_args()
    start_idx = end_idx = None
    if args.interval:
        start_idx, end_idx = args.interval

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
        elif args.loss == 'L1' or args.loss == 'MSE' or args.loss == 'NegPea':
            ref_type = 'PPGSignal'
            print('\nPPGSignal reference type chosen!')
        else:
            ref_type = 'PulseNumerical'
            print('\nPulseNumerical reference type chosen!')

        if args.data == 'UBFC':
            testset = Dataset4DFromUBFC(labels=ref_type,
                                        device=loader_device,
                                        start=start_idx, end=end_idx,
                                        augment=False,
                                        augment_freq=False)
    else:
        print('Error! No such model.')
        exit(666)

    testloader = DataLoader(testset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_cpu,
                            pin_memory=True)

    # --------------------------
    # Load model
    # --------------------------
    model = None
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

    
    model.load_state_dict(tr.load(args.weight), False)

    # Copy model to working device
    model = model.to(device)
    model.eval()

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
        print('\nHey! No such loss function. Choose from: L1, MSE, NegPea, SNR, Gauss, Laplace')
        print('Inference with no loss function')

    # -------------------------------
    # Evaluate model
    # -------------------------------
    eval_model(model, testloader, criterion=loss_fn, oname=args.ofile_name)

    print('Successfully finished!')
