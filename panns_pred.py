import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
sys.path.insert(2,os.path.join(sys.path[0],'utils'))
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np
import torch
import pickle
import glob
from tensorflow.keras.metrics import AUC
from models import Transfer_Cnn14
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
    hop_size, window, pad_mode, center, ref, amin, top_db)

from sklearn.metrics import accuracy_score,f1_score,average_precision_score


def print_audio_tagging_result(clipwise_output,labels=['ashy_prinia','black-faced_antbird','veery','willow_flycatcher']):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    print("C: ",np.exp(clipwise_output))
    # Print audio tagging top probabilities
    for k in range(1):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
            np.exp(clipwise_output[sorted_indexes[k]])))
        ret = np.array(labels)[sorted_indexes[k]]
    return ret



exp_path =  "sounds/explosion_sounds/-0kLxY8r5VY-0.0-7.0.wav"
audio_path = 'XC_Sounds_wav/black-faced_antbird/*'


onehot_labels = []
audio_arrs = []


'''
for audio_path in glob.glob('sounds/explosion_sounds/*.wav'):
    filename = audio_path.split("/")[-1]

    gt_labels = df[df.wav_file==filename]["labels_name"].values
    if len(gt_labels)==0:
        print("File not found: ")
        continue

    onehot = np.zeros(shape=(1,527))

    for gt in gt_labels[0]:
        index = list(labels).index(gt)
        onehot[0][index]=1



    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]
    audio_arrs.append(audio)
    onehot_labels.append(onehot)

'''

print("<--- Audio Tagging ---->")

model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,classes_num, freeze_base=0)
cp = "checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=none/batch_size=1/freeze_base=False/1000_iterations.pth"

at = AudioTagging(model=model,checkpoint_path=cp, device='cuda')
preds = []
bird_name = "black-faced_antbird"
count=0
for ii,filename in enumerate(glob.glob(audio_path)):
    print("Iteration ",ii)
    audio,_ = librosa.load(filename,sr=32000,mono=True)
    audio = audio[None,:]
    (clipwise_output,embedding) = at.inference(audio)

    nm = print_audio_tagging_result(clipwise_output[0])
    if nm==bird_name:
        count+=1

print("Accuracy: ",count/(ii+1))
'''for audio in audio_arrs:
    (clipwise_output, embedding) = at.inference(audio)
    outputs.append(clipwise_output)

outputs = np.concatenate(outputs)
onehot_labels = np.concatenate(onehot_labels)

m = AUC()
m.update_state(onehot_labels,outputs)
print("AUC: ",m.result().numpy())


print("Macro average Precision: ",average_precision_score(onehot_labels,outputs))
print("Micro average precision: ",average_precision_score(onehot_labels,outputs,average="micro"))
outputs[outputs>0.5]=1
outputs[outputs<0.5]=0
print("F1 Micro: ",f1_score(onehot_labels,outputs,average="micro"))


acc_arr = []
for i in range(527):
    acc = accuracy_score(onehot_labels[:,i],outputs[:,i])
    acc_arr.append(acc)
print(sum(acc_arr)/len(acc_arr))'''
