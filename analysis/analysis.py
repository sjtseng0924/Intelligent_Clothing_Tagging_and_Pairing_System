import numpy as np

baseline_path = "rank_baseline.txt"
main_path = "rank_main.txt"

def read_rank(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                k, v = parts
                d[k] = int(v)
    return d

baseline = read_rank(baseline_path)
main = read_rank(main_path)

# 保證順序一致
keys = sorted(set(baseline.keys()) & set(main.keys()), key=int)
baseline_ranks = np.array([baseline[k] for k in keys])
main_ranks = np.array([main[k] for k in keys])

# 統計
main_better = np.sum(main_ranks < baseline_ranks)
baseline_better = np.sum(main_ranks > baseline_ranks)
equal = np.sum(main_ranks == baseline_ranks)
top5_baseline = np.mean(baseline_ranks <= 5)
top5_main = np.mean(main_ranks <= 5)

print(f"Number of Tops Evaluated: {len(keys)}")
print(f"{'Metric':<20}{'Baseline':<12}{'Main Approach'}")
print(f"{'Average Rank':<20}{baseline_ranks.mean():<12.2f}{main_ranks.mean():.2f}")
print(f"{'Top-5 Accuracy':<20}{top5_baseline*100:<12.1f}{top5_main*100:.1f}")
print(f"{'Wins':<20}{main_better}/{len(keys):<8} {baseline_better}/{len(keys):<12}")
# print(f"{'Baseline Wins':<20}{baseline_better:<12}/ {len(keys)}")
# print(f"{'Equal':<20}{equal:<12}/ {len(keys)}")
