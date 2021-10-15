import os
import re

save_path = r'../saved_models/'
static_dict = {}

def get_static(file_name):
    re_bestAcc = re.compile(r'BEST_EVAL_ACC: (([0-9]|\.)*)')  # .group(1)
    re_bestIt = re.compile(r'at ([0-9]*)')  # .group(1)
    re_top1Acc = re.compile(r"eval\ / top - 1 - acc': (([0-9]|\.)*)")
    re_top5Acc = re.compile(r"eval\ / top - 5 - acc': (([0-9]|\.)*)")
    stat = {"bestAcc": 0,
            "bestIt": 0,
            "Top1Acc": [],
            "Top5Acc": [],
            }
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.endswith('iters'):
                stat['bestAcc'] = lines.find(line, re_bestAcc).group(1)
                stat['bestIt'] = lines.find(line, re_bestIt).group(1)
                stat['Top1Acc'].append(lines.find(line, re_top1Acc).group(1))
                stat['Top5Acc'].append(lines.find(line, re_top5Acc).group(1))
    avg_1_1acc = stat['Top1Acc'][-1]
    avg_20_1acc = stat['Top1Acc'][-20:].sum()/20
    avg_50_1acc = stat['Top1Acc'][-50:].sum()/50
    avg_1_5acc = stat['Top5Acc'][-1]
    avg_20_5acc = stat['Top5Acc'][-20:].sum() / 20
    avg_50_5acc = stat['Top5Acc'][-50:].sum() / 50
    return {'Top1_1': avg_1_1acc,
            'Top1_20': avg_20_1acc,
            'Top1_50': avg_50_1acc,
            'Top5_1': avg_1_5acc,
            'Top5_20': avg_20_1acc,
            'Top5_50': avg_50_1acc,
            'BestAcc': stat['bestAcc'],
            'BestIt': stat['bestIt']}

# str = r"[2021-04-13 15:57:33,078 INFO] 228000 iteration, USE_EMA: True, {'train/sup_loss': tensor(0.0311, device='cuda:0'), 'train/unsup_loss': tensor(0.2391, device='cuda:0'), 'train/total_loss': tensor(0.3913, device='cuda:0'), 'train/mask_ratio': tensor(0.5246, device='cuda:0'), 'lr': 0.028670201217471786, 'train/prefecth_time': 0.0050832958221435545,'train/run_time': 0.315829833984375, 'eval/loss': tensor(1.0763, device='cuda:0'), 'eval/top-1-acc': 0.6306},BEST_EVAL_ACC: 0.9348, at 173000 iters"

statics = {}
for name in os.listdir(save_path):
    cur_path = save_path + name
    if os.path.isdir(cur_path):
        cur_name = name
        for n in os.listdir(cur_path):
            if n == 'log.txt':
                statics[cur_name] = get_static(cur_path + '/' + n)

for k, v in statics.items():
    print('------------------------------------')
    print(k)
    for kk, vv in v:
        print(k, '  ', v)
