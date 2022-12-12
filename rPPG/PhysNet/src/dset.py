"""
PyTorch Dataset classes for dataloader
"""
import torch
from torch.utils.data import Dataset
import os
import cv2
import mediapipe as mp
import numpy as np
import h5py
import random
from matplotlib import pyplot
from torchvision.transforms import RandomRotation, ToPILImage, ToTensor, ColorJitter
import torchvision.transforms.functional as TF

from src.utils import img2uint8, pad_to_square
tr = torch

class Dataset4DFromUBFC(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, labels: str, device, start=None, end=None, D=128, C=3, H=128, W=128, 
                 augment=False,  augment_freq=False, ccc=True):
        """
        :param labels: tuple of label names to use (e.g.: ('pulseNumerical', 'resp_signal') or ('pulse_signal', ) )
            Note that the first label must be the pulse rate if it is present!
        :param D: In case of using collate_fn in dataloader, set this to 180 -> D=180
        :param ccc: color channel centralization
        """
        self.device = device
        self.D = D
        self.H = H
        self.W = W
        self.C = C
        self.augment = augment
        self.augment_frq = augment_freq
        self.ccc = ccc

        # ---------------------------
        # Augmentation variables
        # ---------------------------
        self.flip_p = None
        self.rot = None
        self.color_transform = None
        self.freq_scale_fact = None

        # -----------------------------
        # Open database
        # -----------------------------

        # dataset folder
        root = '/home/kuangjie/dataset/UBFC/'

        # get folder/subject list
        subjects = os.listdir(root)

        # init labels and frames
        self.frames = np.empty((16128, 128, 128, 3), dtype = 'uint8')
        # self.frames = np.empty((60032, 128, 128, 3), dtype = 'uint8')
        # self.frames = np.empty((79369, 480, 640, 3), dtype = 'uint8')
        framenum = 0
        labels_PulseNumerical = []
        labels_PPGSignal = []

        mp_face_detection = mp.solutions.face_detection

        # iterate through all directories
        for i in range(len(subjects)):
        # for i in range(32):
            if i > 31:
                subjectpath = root + subjects[i]

                # load ground truth
                gtfilepath = subjectpath + '/ground_truth.txt'
                if os.path.isfile(gtfilepath):
                    gtdata = np.loadtxt(gtfilepath)
                    gtTrace = gtdata[0,:]
                    gtTime = gtdata[2,:]
                    gtHR = gtdata[1,:]
                
                # normalize data (zero mean and unit variance)
                gtTrace = gtTrace - np.mean(gtTrace)
                gtTrace = gtTrace / np.std(gtTrace, ddof = 1)

                maxlabelcount = 128 * (len(gtTrace) // 128)

                labels_PulseNumerical.extend(gtHR[0:maxlabelcount])
                labels_PPGSignal.extend(gtTrace[0:maxlabelcount])

                # load video
                vidfilepath = subjectpath + '/vid.avi'
                vid = cv2.VideoCapture(vidfilepath)

                # video basic info
                # framerate  = vid.get(cv2.CAP_PROP_FPS)
                framecount  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                # framewidth  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                # frameheight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # print(gtdata.shape, framerate, framecount)
                
                count = 0
                ret = True
                bboxC = None
                maxcount = 128 * (framecount // 128)
                
                while(count < maxcount and ret):
                    
                    ret, frame = vid.read()
                    if (count % 128) == 0:
                        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            # Draw face detections of each face.
                            if not results.detections:
                                print('count:{count} no face')
                                continue
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                            h,w,c = frame.shape
                            face = frame[int(bboxC.ymin*h):int(bboxC.ymin*h+bboxC.height*h-1),int(bboxC.xmin*w):int(bboxC.xmin*w+bboxC.width*w-1),:]
                            face = cv2.resize(face, (self.H, self.W), interpolation=cv2.INTER_AREA)
                            face = img2uint8(face)
                            self.frames[framenum] = face
                    else:
                        h,w,c = frame.shape
                        face = frame[int(bboxC.ymin*h):int(bboxC.ymin*h+bboxC.height*h-1),int(bboxC.xmin*w):int(bboxC.xmin*w+bboxC.width*w-1),:]
                        face = cv2.resize(face, (self.H, self.W), interpolation=cv2.INTER_AREA)
                        face = img2uint8(face)
                        self.frames[framenum] = face
                    count += 1
                    framenum += 1
                print(maxcount)
                print('Done: ',i)
        
        print('Frames and labels are ready!')

        # Append all required label from database
        self.label_names = labels
        if labels == 'PulseNumerical':
            self.labels = np.array(labels_PulseNumerical)
        elif labels == 'PPGSignal':
            self.labels = np.array(labels_PPGSignal)
        else:
            print('Wrong labels! Please choose from PulseNumerical and PPGSignal.')

        (self.n, H, W, C) = self.frames.shape
        print(f'Number of frames in the whole dataset: {self.n}')
        print(len(self.labels))
        print(framenum)
        if start is not None:
            self.n = end - start
            self.begin = start
        else:
            self.begin = 0

        print(f'Number of images in the chosen interval: {self.n}')
        print(f'Size of an image: {H} x {W} x {C}')

        self.num_samples = ((self.n - 64) // self.D) - 1
        print(f'Number of samples in the dataset: {self.num_samples}\n')
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.D
        
        # ------------------------------------
        # Set up video augmentation parameters
        # ------------------------------------
        if self.augment:
            # Set up the same image transforms for the chunk
            self.flip_p = random.random()
            self.hflip_p = random.random()
            self.rot = RandomRotation.get_params((0, 90))
            self.color_transform = ColorJitter.get_params(brightness=(0.7, 1.3),
                                                          contrast=(0.8, 1.2),
                                                          saturation=(0.8, 1.2),
                                                          hue=(0, 0))

        # -------------------------------
        # Set up frequency augmentation
        # -------------------------------
        if self.augment_frq:
            self.freq_scale_fact = np.around(np.random.uniform(0.7, 1.4), decimals=1)
            d = int(np.around(self.D * self.freq_scale_fact))

        # ---------------------------
        # Construct target signals
        # ----------------------------
        # print(self.labels.shape)
        # print(self.labels)
        label_segment = self.labels[self.begin + idx * self.D: self.begin + idx * self.D + self.D]
        label_segment = np.array(label_segment)
        label_segment = tr.from_numpy(label_segment).type(tr.FloatTensor)
        # If numerical select mode value
        if self.label_names == 'PulseNumerical':
            # label_segment = tr.mode(label_segment.squeeze())[0] / 60.
            # label_segment = tr.mean(label_segment.squeeze()) / 60.
            label_segment = label_segment.squeeze() / 60.
        targets = label_segment
        # print(targets.shape)
        
        # ----------------------------
        # Construct networks input
        # -----------------------------
        video = tr.empty(self.C, d, self.H, self.W, dtype=tr.float)


        # -------------------------------
        # Fill video with frames
        # -------------------------------
        # conv3d input: N x C x D x H X W
        for i in range(d):
            img = self.frames[self.begin + idx * self.D + i, :]

            # if i % 32 == 0 and idx == 0:
            #     cv2.imwrite('/home/kuangjie/code/kuangjie-framework/infer_img' + str(i) +'.png',img)
            
            
            # Augment if needed and transform to tensor
            if self.augment:
                img = self.img_transforms(img)
            else:
                img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W
                if self.ccc:
                    img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization

            video[:, i, :] = img

        # ------------------------------
        # Apply frequency augmentation
        # ------------------------------
        # if self.augment_frq:
        #     sample = self.freq_augm(d, idx, targets, video)
        # else:
        #     if self.ccc:
        #         # targets = np.array(targets)
        #         # targets = tr.from_numpy(targets).type(tr.FloatTensor)
        #         targets = tr.mean(targets)
        #     sample = (video, targets)
        
        targets = np.array(targets)
        # if idx == 0:
        #     pyplot.plot(targets, 'k-')
        #     pyplot.title('test_targets')
        #     pyplot.savefig('/home/kuangjie/code/kuangjie-framework/infer_targets.png')
        targets = tr.from_numpy(targets).type(tr.FloatTensor)
        sample = (video, targets)

        # Video shape: C x D x H X W
        # print(targets.shape)
        # print(targets)
        

        return sample

    def collate_fn(self, batch):
        """
        This function applies the same augmentation for each batch to result in an LSTM sequence
        """
        videos, targets = list(zip(*batch))

        # Set up the same image transforms for the given number of batches
        self.flip_p = random.random()
        self.hflip_p = random.random()
        self.color_transform = ColorJitter.get_params(brightness=(0.5, 1.3),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(0, 0))

        # set up parameters for frequency augmentation
        desired_d = 128
        self.freq_scale_fact = np.around(np.random.uniform(0.7, 1.3), decimals=1)
        d = int(np.around(desired_d * self.freq_scale_fact))

        # -----------------------------
        # Augment labels accordingly
        # -----------------------------
        targets = tr.stack([tr.mean(target[:d]) * self.freq_scale_fact for target in targets]).unsqueeze(1)
        # print(f'Targets: {targets.shape}')

        # -------------------------------------
        # Augment video same way for each batch
        # -------------------------------------
        # Frequency augmentation
        # print(len(videos))
        videos = tr.stack(videos)
        # print(f'videos: {videos.shape}')
        resampler = torch.nn.Upsample(size=(desired_d, self.H, self.W), mode='trilinear', align_corners=False)
        videos = resampler(videos[:, :, 0:desired_d, :, :])
        # Image augmentation
        for b in range(videos.shape[0]):
            for d in range(videos.shape[2]):
                videos[b, :, d, :, :] = self.img_transforms(videos[b, :, d, :, :])

        return videos, targets

    def freq_augm(self, d, idx, targets, video):
        # edit video
        resampler = torch.nn.Upsample(size=(self.D, self.H, self.W), mode='trilinear', align_corners=False)
        video = resampler(video.unsqueeze(0)).squeeze()
        # edit labels
        if self.label_names == 'PulseNumerical':
            targets = targets * self.freq_scale_fact
        elif self.label_names == 'PPGSignal':
            segment = tr.from_numpy(
                self.labels[self.begin + idx * self.D: self.begin + idx * self.D + d])
            resampler = torch.nn.Upsample(size=(self.D,), mode='linear', align_corners=False)
            segment = resampler(segment.view(1, 1, -1))
            segment = segment.squeeze()
            targets = segment
        sample = ((video,), *targets)
        return sample

    def img_transforms(self, img):
        img = ToPILImage()(img)
        if self.flip_p > 0.5:
            img = TF.vflip(img)
        if self.flip_p > 0.5:
            img = TF.hflip(img)
        # img = TF.rotate(img, self.rot)
        img = self.color_transform(img)
        img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W

        img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization
        return img

    def bbox_checker(self, x1, x2, y1, y2):
        # check to be inside image size
        if y2 > self.H:
            y2 = self.H
        if x2 > self.W:
            x2 = self.W
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        # check validity
        if y2 - y1 < 1 or x2 - x1 < 1:
            y1 = x1 = 0
            y2, x2 = self.W, self.H
        return x1, x2, y1, y2
