from collections import defaultdict
import torch
import sentencepiece as spm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd


# location = "./analysis_tokens_v2/4_2,2,2,2/"
# language_pairs = 'nso-eng,msa-eng,tgl-eng,cat-eng,eng-nso,eng-msa,eng-tgl,eng-cat'.split(",")
location = "./analysis/"
langs = 'ind,msa,isl,slv,tgl,nso,run,nob,cat,glg,ssw,fao,fur,ltz,lim'.split(",")
language_pairs = []
for lg in langs:
    language_pairs.append(f"{lg}-eng")
for lg in langs:
    language_pairs.append(f"eng-{lg}")
h_layer = 4
NUM1=0
NUM2=6
# ## combine all of them
# stats = {}
# for lg_pair in language_pairs:
#     stats[lg_pair] = [defaultdict(lambda: [[0]*(h_layer+1), [0]*(h_layer+1)]) for _ in range(6)]
#     for gpu_id in range(32):
#         info = torch.load(location + lg_pair + "_2" +str(gpu_id))
#         for hid in range(6):
#             tokens = sorted(info[hid], key=lambda x: info[hid].get(x)[1][h_layer-1], reverse=True)
#             for token in tokens:
#                 v = info[hid][token]
#                 v0 = np.array(v[0])
#                 v1 = np.array(v[1])
#                 n0 = np.array(stats[lg_pair][hid][token][0])
#                 n1 = np.array(stats[lg_pair][hid][token][1])
#                 for i in range(len(v0)):
#                     if v1[i] + n1[i] !=0:
#                         stats[lg_pair][hid][token][0][i] = (n0[i] * n1[i] + v0[i] * v1[i]) / (n1[i] + v1[i])
#                         stats[lg_pair][hid][token][1][i] = n1[i] + v1[i]
#                     else:
#                         assert v0[i] == 0 or n[i] == 0
#                 # stats[lg_pair][hid][token][0] = stats[lg_pair][hid][token][0].tolist()
#                 # stats[lg_pair][hid][token][1] = stats[lg_pair][hid][token][1].tolist()
# for lg_pair in language_pairs:
#     for i in range(len(stats[lg_pair])):
#         stats[lg_pair][i] = dict(stats[lg_pair][i])
# torch.save(stats, location + "all_stats2")

## Analysis
#1: lg-pair, hid
# stats = torch.load(location + "all_stats")
# avg_layer = [defaultdict(lambda: [np.array([0]*(h_layer+1)), np.array([0]*(h_layer+1))]) for _ in range(NUM1, NUM2)]
# for lg_pair in language_pairs:
#     for hid in range(NUM1, NUM2):
#         for token, v in stats[lg_pair][hid].items():
#             v0 = np.array(v[0])
#             v1 = np.array(v[1])
#             n0 = avg_layer[hid][lg_pair][0]
#             n1 = avg_layer[hid][lg_pair][1]
#             avg_layer[hid][lg_pair][0] = (n0 * n1 + v0 * v1) / (n1 + v1)
#             avg_layer[hid][lg_pair][1] = n1 + v1
# for hid in range(NUM1, NUM2):
#     for lg_pair in language_pairs:
#         print(f"{lg_pair}, {hid}, {avg_layer[hid][lg_pair][0]}")
    
#2: lg-pair
# stats = torch.load(location + "all_stats")
# avg_layer = defaultdict(lambda: [np.array([0]*(h_layer+1)), np.array([0]*(h_layer+1))])
# for lg_pair in language_pairs:
#     for hid in range(NUM1,NUM2):
#         for token, v in stats[lg_pair][hid].items():
#             v0 = np.array(v[0])
#             v1 = np.array(v[1])
#             n0 = avg_layer[lg_pair][0]
#             n1 = avg_layer[lg_pair][1]
#             avg_layer[lg_pair][0] = (n0 * n1 + v0 * v1) / (n1 + v1)
#             avg_layer[lg_pair][1] = n1 + v1
# for lg_pair in language_pairs:
#     print(f"{lg_pair}, {avg_layer[lg_pair][0]}")

# #2: h_id
# stats = torch.load(location + "all_stats")
# avg_layer = [[np.array([0]*(h_layer+1)), np.array([0]*(h_layer+1))] for _ in range(NUM1, NUM2)]
# for lg_pair in language_pairs:
#     for hid in range(NUM1, NUM2):
#         for token, v in stats[lg_pair][hid].items():
#             v0 = np.array(v[0])
#             v1 = np.array(v[1])
#             n0 = avg_layer[hid][0]
#             n1 = avg_layer[hid][1]
#             avg_layer[hid][0] = (n0 * n1 + v0 * v1) / (n1 + v1)
#             avg_layer[hid][1] = n1 + v1
# for hid in range(NUM1, NUM2):
#     print(f" {hid}, {avg_layer[hid][0]}")


## low,high vs. rank
# sns.set(font_scale=1)
f_tokens = torch.load("analysis/m15")
sorted_freq_tokens = sorted(f_tokens, key=f_tokens.get, reverse=True)
freq_rank = {k:i for i, k in enumerate(sorted_freq_tokens)}

stats = torch.load(location + "all_stats")
lowf = []
highf = []
seen_h = set()
seen_l = set()
for lg_pair in language_pairs:
    for hid in range(NUM1, NUM2):
        # print(f" {lg_pair}-------layer == {hid}----------------------------")
        keys_high = sorted(stats[lg_pair][hid], key = lambda x: stats[lg_pair][hid].get(x)[0][-1], reverse=True)
        keys_low = sorted(stats[lg_pair][hid], key = lambda x: stats[lg_pair][hid].get(x)[0][-1], reverse=False)
        for k in keys_high[:25]:
            if k in freq_rank and stats[lg_pair][hid].get(k)[0][-1] == 4:
                highf.append(freq_rank[k])
                # if freq_rank[k] > 30000: 
                    
                # print(lg_pair, k, stats[lg_pair][hid].get(k)[0][-1], freq_rank[k], "high")
                seen_h.add(k)
        for k in keys_low[:25]:
            if k in freq_rank and stats[lg_pair][hid].get(k)[0][-1] == 1:
                lowf.append(freq_rank[k])
                # if freq_rank[k] < 2000:  
                    # print(lg_pair, k, stats[lg_pair][hid].get(k)[0][-1], freq_rank[k], "low")
                seen_l.add(k)
            # print(k, stats[lg_pair][hid][k][0], freq_rank.get(k, -1))
print(f"len of high {len(highf)}, {len(seen_h)}")
print(f"len of low {len(lowf)},  {len(seen_l)}")

# c_f = []
# for word in seen_h:
#     if word in freq_rank:
#         c_f.append([word, freq_rank[word]])
# c_f = sorted(c_f, key=lambda x: x[1])
# for w, f in c_f:
#     print(w, f)


data = pd.DataFrame({
    "Token Frequency Rank": highf + lowf,
    "": ["High Requested Capacity"] * len(highf) + ["Low Requested Capacity"] * len(lowf)
})

plt.figure(figsize=(12, 6))
b=sns.violinplot(x="",y="Token Frequency Rank", data=data, color="skyblue", fontsize=16)
b.set_yticklabels(b.get_yticks(), size = 14)
plt.xlabel('', fontsize=16)
plt.ylabel('Token Frequency Rank', fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
plt.savefig("./analysis/tmp.pdf")



## rank vs. #pass

# f_tokens = torch.load("analysis/m15")
# sorted_freq_tokens = sorted(f_tokens, key=f_tokens.get, reverse=True)
# high_freq = sorted_freq_tokens[:2000:50]
# low_freq = sorted_freq_tokens[-7500:]
# # print(high_freq)

# # exit(0)
# stats = torch.load(location + "all_stats")
# lowf = []
# highf = []
# for lg_pair in language_pairs:
#     for hid in range(NUM1, NUM2):
#         # print(f" {lg_pair}-------layer == {hid}----------------------------")
#         for k in high_freq:
#             if k in stats[lg_pair][hid]:
#                 highf.append(stats[lg_pair][hid][k][0][-1])
#         for k in low_freq:
#             if k in stats[lg_pair][hid]:
#                 lowf.append(stats[lg_pair][hid][k][0][-1])
#             # print(k, stats[lg_pair][hid][k][0], freq_rank.get(k, -1))
# print(f"len of high {len(highf)}")
# print(f"len of low {len(lowf)}")
# data = pd.DataFrame({
#     "Num of Passes": highf + lowf,
#     "": ["High Freq Tokens"] * len(highf) + ["Low Freq Tokens"] * len(lowf)
# })

# plt.figure(figsize=(12, 6))
# sns.violinplot(x="",y="Num of Passes", data=data, palette="Set3")
# plt.show()
# plt.savefig("./analysis/tmp2.png")
