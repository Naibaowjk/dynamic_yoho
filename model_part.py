# create the part model & show result
import torch
from mobilenet import mobilenet_19
from mobilenet_part import build_part
import os
import numpy as np
import torch.nn as nn
import librosa
import time
from sklearn import preprocessing
np.random.seed(42)


print(os.getcwd())

model = mobilenet_19(num_emed=256, n_spk=4, mode='inference')
model_path = "./model_params.pkl"
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
# model = model.cuda()
model.eval()
model_part1, model_part2, model_part3 = build_part(model, mode="inference")
model_part1.eval()
model_part2.eval()
model_part3.eval()

# prepare data
def load_data(path_list):
    scaler = preprocessing.MaxAbsScaler()
    sources = []
    mix = np.zeros(160000, dtype=np.float32)
    for path in path_list:
        data, _ = librosa.load(path, sr=16000, mono=True)
        mix += data
        data = scaler.fit_transform(data.reshape(-1, 1)).T
        sources.append(data)
    mix = scaler.fit_transform(mix.reshape(-1, 1)).T
    return mix
    
s1_normal = "./data/normals/pump/0.wav"
s2_normal = "./data/normals/slider/0.wav"
s3_normal = "./data/normals/fan/0.wav"
s4_normal = "./data/normals/valve/0.wav"
s1_abnormal = "./data/abnormals/pump/0.wav"
s2_abnormal = "./data/abnormals/slider/0.wav"
s3_abnormal = "./data/abnormals/fan/0.wav"
s4_abnormal = "./data/abnormals/valve/0.wav"

# 160000
mix_baseline = load_data([s1_normal, s2_normal, s3_normal, s4_normal])
mix = load_data([s1_abnormal, s2_normal, s3_normal, s4_abnormal])
# 98304
mix_baseline_short = mix_baseline[:, :98304]
mix_short = mix[:, :98304]

print(mix_baseline.shape)
print(mix.shape)
print(mix_baseline_short.shape)
print(mix_short.shape)

# conver numpy to Tensor
mix_baseline_short = torch.Tensor(mix_baseline_short).to("cpu")
mix_short = torch.Tensor(mix_short).to("cpu")

# compute results of entire ia-net-lite
print(mix_baseline_short.size())
print(mix_short.size())

# compute results of distributed ia-net-lite
t_start = time.perf_counter()
features_dis = model_part1(mix_short.unsqueeze(0))
features_dis = model_part2(features_dis)
features_dis = model_part3(features_dis)
t_dis = time.perf_counter() - t_start
print("Dis. computation time is {}".format(t_dis))