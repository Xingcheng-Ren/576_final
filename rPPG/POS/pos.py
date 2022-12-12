import os
import cv2
import json
import h5py
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib import pyplot

# -----------------------------
# Init Mediapipe Face Detection
# -----------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def bbox_calculate(frame):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results

def annotated_image(frame, results):
    annotated_image = frame.copy()
    for detection in results.detections:
        mp_drawing.draw_detection(annotated_image, detection)
    return annotated_image

def face_extract(frame, results):
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        h,w,_ = frame.shape
    face = frame[int(bboxC.ymin*h):int(bboxC.ymin*h+bboxC.height*h-1), \
                 int(bboxC.xmin*w):int(bboxC.xmin*w+bboxC.width*w-1)]
    return face

def meanrgb_calculate(face):
    h,w,_ = face.shape
    num_face_pixels = h*w
    mean_r = np.sum(face[:,:,2])/num_face_pixels
    mean_g = np.sum(face[:,:,1])/num_face_pixels 
    mean_b = np.sum(face[:,:,0])/num_face_pixels
    return [mean_r,mean_g,mean_b]

def fft_calculate(signal, framerate, n):
    minFreq = 0.5
    maxFreq = 8
    signal = signal - np.mean(signal)
    fft_data = np.fft.rfft(signal, n) # FFT
    fft_data = np.abs(fft_data)
    freq = np.fft.rfftfreq(n, 1.0/framerate) # Frequency data
    inds= np.where((freq < minFreq) | (freq > maxFreq))[0]
    fft_data[inds] = 0
    max_index = np.argmax(fft_data)
    # hr = framerate / 2048 * max_index * 60
    hr = freq[max_index] * 60
    return fft_data, hr

def pos(signal, framerate):
    l = int(framerate*3.2) # window length
    length = signal.shape[1]
    H = np.zeros(length) # heart trace

    for t in range(0, (length-l+1)):
        C = signal[:,t:t+l-1]
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv,C)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        P = np.matmul(std,S)
        H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))

    return H

def UBFC():
    D = 128
    final_hr = []
    final_gt = []
    
    # dataset folder
    root = '/home/kuangjie/dataset/UBFC/'

    # get folder/subject list
    subjects = os.listdir(root)

    # iterate through all directories
    for i in range(len(subjects)):
    # for i in range(2):
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
        target  = gtTrace

        # load video
        vidfilepath = subjectpath + '/vid.avi'
        video = cv2.VideoCapture(vidfilepath)

        # video basic info
        # framerate  = video.get(cv2.CAP_PROP_FPS)
        framerate   = 30.0
        framecount  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        framewidth  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(gtdata.shape, framerate, framecount)

        sample = framecount // D
        maxcount = D * sample
        
        count = 0
        ret = True
        results = None
        mean_rgb = np.empty((3, maxcount))
        while(count < maxcount and ret):
            ret, frame = video.read()
            if count % D == 0:
                results = bbox_calculate(frame)
                if not results.detections:
                    print('count:{count} no face')
                    continue
            face = face_extract(frame, results)
            mean_rgb[:,count] = meanrgb_calculate(face)
            count += 1
        
        signal = pos(mean_rgb, framerate) # POS
        signal = signal.flatten() 
        
        # pyplot.plot(range(signal.shape[0]), signal, 'k-')
        # pyplot.title('Pulse Signal POS')
        # pyplot.show()

        for c in range(sample):
            signal_segm = signal[c * D : c * D + D]
            target_segm = target[c * D : c * D + D]
            spectrum_output, output_hr = fft_calculate(signal_segm, framerate, 2048)
            spectrum_target, target_hr = fft_calculate(target_segm, framerate, 2048)
            final_hr.append(output_hr)
            final_gt.append(target_hr)

            fig, axes = pyplot.subplots(2, 2, figsize=(18, 18))
            axes[0,0].plot(signal_segm)
            axes[0,1].plot(spectrum_output)
            axes[1,0].plot(target_segm)
            axes[1,1].plot(spectrum_target)
            axes[0,0].set_title(str(output_hr))
            axes[1,0].set_title(str(target_hr))
            pyplot.savefig('pos-ubfc/ubfc' + str(i) + '-' + str(c) + ".png")

    df = pd.DataFrame({'ground_truth': final_gt, 'heart_rate': final_hr})
    df.to_excel('/home/kuangjie/code/mycode/pos_ubfc_new.xlsx', sheet_name='POS', index=False)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest = 'dataset', help='UBFC, PURE, COHFACE, MAHNOB-HCI')
    args = parser.parse_args()
    if args.dataset == 'UBFC':
        UBFC()
         
if __name__ == '__main__':
	main()