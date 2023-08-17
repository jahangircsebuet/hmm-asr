import gc

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle as pkl
import librosa
from scipy.io import wavfile
from tqdm import tqdm, trange
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         pass
#     break
    # print(os.path.join(dirname, filename))

path = 'data'
data_path = path + '/data'
print(path, data_path)


def getSentenceDict():
    sentence_dict = {}
    with open(path + "/PROMPTS.TXT") as fr:
        line = fr.readline()
        while line:
            if (line[0] != ';'):
                line = line.split('(')
                s_id = line[1].split(')')[0]
                s_id = s_id.lower()
                sentence_dict[s_id] = line[0]
            line = fr.readline()
    return sentence_dict


phon61_map39 = {
    'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ix': 'ih', 'ax': 'ah', 'ah': 'ah', 'uw': 'uw',
    'ux': 'uw', 'uh': 'uh', 'ao': 'aa', 'aa': 'aa', 'ey': 'ey', 'ay': 'ay', 'oy': 'oy', 'aw': 'aw',
    'ow': 'ow', 'l': 'l', 'el': 'l', 'r': 'r', 'y': 'y', 'w': 'w', 'er': 'er', 'axr': 'er',
    'm': 'm', 'em': 'm', 'n': 'n', 'nx': 'n', 'en': 'n', 'ng': 'ng', 'eng': 'ng', 'ch': 'ch',
    'jh': 'jh', 'dh': 'dh', 'b': 'b', 'd': 'd', 'dx': 'dx', 'g': 'g', 'p': 'p', 't': 't',
    'k': 'k', 'z': 'z', 'zh': 'sh', 'v': 'v', 'f': 'f', 'th': 'th', 's': 's', 'sh': 'sh',
    'hh': 'hh', 'hv': 'hh', 'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'qcl': 'sil', 'bcl': 'sil', 'dcl': 'sil',
    'gcl': 'sil', 'h#': 'sil', '#h': 'sil', 'pau': 'sil', 'epi': 'sil', 'nx': 'n', 'ax-h': 'ah', 'q': 'sil'
}
phon61 = list(phon61_map39.keys())
phon39 = list(set(phon61_map39.values()))

label_p39 = {}
p39_label = {}
for i, p in enumerate(phon39):
    label_p39[p] = i + 1
    p39_label[i + 1] = p

phon39_map61 = {}
for p61, p39 in phon61_map39.items():
    if not p39 in phon39_map61:
        phon39_map61[p39] = []
    phon39_map61[p39].append(p61)


# ------------------------------------------------------------------------

def get39EquiOf61(p):
    return phon61_map39[p]


def removePhonStressMarker(phon):
    phon = phon.replace('1', '')
    phon = phon.replace('2', '')
    return phon


def readTrainingDataDescriptionCSV():
    file_path = path + '/train_data.csv'  # check if train_data.csv is in correct path
    tdd = pd.read_csv(file_path)
    # removing NaN entries in the train_data.csv file
    dr = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
    tdd = tdd[tdd['dialect_region'].isin(dr)]
    return tdd


def readTestingDataDataDescriptionCSV():
    file_path = path + '/test_data.csv'  # check if train_data.csv is in correct path
    tdd = pd.read_csv(file_path)
    # removing NaN entries in the train_data.csv file
    dr = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
    tdd = tdd[tdd['dialect_region'].isin(dr)]
    return tdd


# data_reader
def data_reader(read='train', dump_pickle=False, max_len=None):
    data = {}
    # label = []

    # -------------------------------------
    file_path = data_path + "/"
    f_Path = 'path_from_data_dir'  # field that contains file path in train_data.csv
    f_IsAudio = 'is_converted_audio'  # boolean field that tells that the record in train_data.csv contains the description of audio file we are interested in
    f_IsWord = 'is_word_file'
    f_IsPhon = 'is_phonetic_file'
    f_IsSent = 'is_sentence_file'
    # f_filename = 'filename' #field that contains filename
    f_dr = 'dialect_region'  # field that contains dialect_region information

    # -----------------------------------------
    if read == 'train':
        tdd = readTrainingDataDescriptionCSV()
    elif read == 'test':
        tdd = readTestingDataDataDescriptionCSV()
    else:
        raise Exception('"read" paramater can only assume "train" or "test"')

    afd = tdd[tdd[f_IsAudio] == True]  # audio file descriptions list
    pfd = tdd[tdd[f_IsPhon] == True]  # phone file descriptions list
    del (tdd)

    # -----------------------------------------
    max_ = {}
    p_bar = tqdm(range(100))
    c = -1

    num = 0
    # start of data and max_ generation
    for i, j in afd.iterrows():
        if num < 2:
            num = num + 1
            c += 1
            afp = file_path + j[f_Path]  # audio file path
            pfn = j['filename'].split('.WAV')[0] + '.PHN'

            p_bar.set_description(f'Working on {j["filename"]} ,index: {c}  ')
            try:
                pfp = file_path + pfd[(pfd['filename'] == pfn) & (pfd['speaker_id'] == j['speaker_id'])][f_Path].values[0]
            except:
                pfp = afp.replace(j['filename'], pfn)

            # print(pfp)
            ph_ = pd.read_csv(pfp, sep=" ")
            # assign column name
            ph_.columns = ['start', 'end', 'phoneme']
            # audio,sr = librosa.load(afp,sr=None)
            sr, audio = wavfile.read(afp)

            # all of the data didn't fit in the main memory so trying different solution
            for k, l in ph_.iterrows():
                # what is difference between 39 and 61
                label = get39EquiOf61(removePhonStressMarker(l['phoneme']))
                if label not in data:
                    data[label] = []
                data[label].append(audio[l['start']:l['end']])
                if (label not in max_):
                    max_[label] = 0
                else:
                    s = l['end'] - l['start']
                    max_[label] = sr if s > sr else s if max_[label] < s else max_[label]

    # end of data and max_ generation

    if (max_len is not None):
        max_ = max_len

    # iterate over data
    for key, dat in data.items():
        dt = np.zeros((len(dat), max_[key]))
        print(dt.shape)
        for i in range(dt.shape[0]):
            try:
                dt[i][:data[key][i].shape[0]] = data[key][i]
            except:
                dt[i] = data[key][i][:max_[key]]
        data[key] = dt
        '''
            label.append(label_p39[get39EquiOf61(removePhonStressMarker(l['phoneme']))])
            data.append(audio[l['start']:l['end']])
        '''
    print(data)
    p_bar.close()
    if (dump_pickle and False):
        output_path = '/output/train_data.pickle' if read == 'train' else '/output/test_data.pickle'
        fo = open(output_path, 'wb')
        pkl.dump({"data": np.array(data, dtype=object)}, fo)
        fo.close()

    # Because 4-5 separate phoneme in TIMIT 61 is labeled as 'sil' in 39 phoneme there are many 'sil' data
    # it took very long time to train for 'sil' thus only using half of the data
    data['sil'] = data['sil'][:int(data['sil'].shape[0] / 2)]
    return data, max_


# ------------------------------------------------------------------------------

data, max_ = data_reader()
print(data['sil'].shape)

from hmmlearn import hmm
import random
from librosa import display
import matplotlib.pyplot as plt
import time

# Preparing the data and training
# --------------------------------------------

models = {}
# gc.collect()


# print(librosa.load(data_path+"/"+"TRAIN/DR1/FCJF0/SA1.WAV.wav",sr=None))
def train_model(data):
    # computing mfcc's
    train_data = []
    c = 0
    print("Computing MFCC features")
    for x in data:
        mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13)

        if (len(train_data) > 0):
            train_data = np.concatenate([train_data, mfcc])
        else:
            train_data = mfcc
        '''
        if(c<2 or False):
            pass
            m_delta = librosa.feature.delta(mfcc)
            m_delta_delta = librosa.feature.delta(m_delta)
            fig, ax = plt.subplots(1,3)
            img1 = librosa.display.specshow(mfcc, x_axis='time', ax=ax[0])
            img2 = librosa.display.specshow(m_delta, x_axis='time', ax=ax[1])
            img3 = librosa.display.specshow(m_delta_delta, x_axis='time', ax=ax[2])

            ax[0].set(title='MFCC')
            ax[1].set(title='DELTA')
            ax[2].set(title='DELTA DELTA')
            c+=1
        '''

    print('Training Feature Data Collected')
    model = hmm.GMMHMM(n_components=13, n_mix=1)
    model.fit(train_data)
    print('Training Complete')
    del (train_data)
    return model  # ,train_data


# training --- and saving training_data's
t = time.process_time()
print(0, 'time elapsed\n------------\n')
for i, ph_ in enumerate(phon39):
    # gc.collect()
    print("training HMM model for", ph_)
    if True:  # ph_ not in ['p','g','hh','sil','dx','r','ih','s','sh','oy','v','uh','ch']:#ph_+".pickle" not in os.listdir('/kaggle/working'):
        model = train_model(data[ph_])
        models[ph_] = model
        # print("saving training data into pickle")
        # output_path = '/kaggle/working/'+ph_+".pickle"
        # fo = open(output_path,'wb')
        # pkl.dump(train_data, fo)
        # fo.close()
        # print('saved')
    t = time.process_time() - t
    print(t, 'time elapsed\n------------\n')

test_data, max_ = data_reader(read='test', max_len=max_)


def testModel(data_phon_label, models, data, max_):
    predictions = []
    correct = 0
    print("testing ", data_phon_label, "\n")
    for x in data:
        s = models[data_phon_label].score(librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13))
        s = 0
        p = s > 0
        for k in max_:
            if (k != data_phon_label):
                tmp = np.zeros(max_[k])
                if max_[k] > max_[data_phon_label]:
                    tmp[:max_[data_phon_label]] = x
                    tmp[max_[data_phon_label]:max_[k]] = x[max_[k] - max_[data_phon_label]]
                else:
                    tmp[:] = x[:max_[k]]
                print(tmp.shape, max_[k])
                o = models[k].score(librosa.feature.mfcc(y=tmp, sr=16000, n_mfcc=13))
                p = s > o
                print(s, o, p)
        if p:
            correct += 1

        predictions.append(p)
    accuracy = (correct / data.shape[0]) * 100
    print(f"No.of Correct Predictions: {correct}\naccuracy: {accuracy}%")
    return correct, accuracy


#     print('Phoneme Recognition ------')
#     predictions = model.score(test_data)
#     del(data)
#     print('Testing Complete')
#     del(test_data)
#     print(predictions)
#     return predictions#,train_data -

def test_all_models(models, test_data):
    pass


predictions = {}
for key in test_data:
    predictions[key] = testModel(key, models, test_data[key], max_)





