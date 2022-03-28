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
from spliter import Spliter
from combiner import Combiner
from mobilenet_part_spilt import Part_Conv_3, Part_FC_3
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
print(mix_short.shape)



# start running
part_3_conv = Part_Conv_3(model, mode="inference")
part_3_fc = Part_FC_3(model, mode="inference")

start_time = time.time()

s1 = Spliter(model_part1, 98304, 256, 32)
ans, ans_time= s1.split_and_compute(mix_short)
split_list = print(len(ans))
c2 = Combiner(model_part2, -1, -1, 2)
ans_2 = []
ans_time2 = []
count = 0
for x in ans:
    out, time_run= c2.combine_and_compute(x)
    if out is not None:
        ans_time2.append(time_run)
        count += 1
        ans_2.append(out)
print(len(ans_2))
c3 = Combiner(part_3_conv, -1, -1, 8)
ans_3 = []
ans_time3= []
for x in ans_2:
    out, time_run= c3.combine_and_compute(x)
    if out is not None:
        ans_3.append(out)
        ans_time3.append(time_run)
print(ans_3[0].size())
print(len(ans_3))
feature_combine_2 = torch.cat(ans_3, dim=2)
print(feature_combine_2.size())
feature_combine_2 = part_3_fc(feature_combine_2)


end_time = time.time()
print(f"total time: {end_time - start_time}")

print(f" split running model use : {sum(ans_time)}")
print(f"combine2 running model use: {sum(ans_time2)}")
print(f"combine3 running model use: {sum(ans_time3)}")
print(f"sum model use {sum(ans_time)+sum(ans_time2)+sum(ans_time3)}")