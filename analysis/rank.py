import matplotlib.pyplot as plt

# 讀取排名
def read_rank(path):
    d = {}
    with open(path) as f:
        for line in f:
            k, v = line.strip().split()
            d[k] = int(v)
    return d

baseline = read_rank('rank_baseline.txt')
main = read_rank('rank_main.txt')

# 保證順序一致
keys = sorted(baseline.keys(), key=int)
baseline_ranks = [baseline[k] for k in keys]
main_ranks = [main[k] for k in keys]

plt.figure(figsize=(12,6))
plt.plot(keys, baseline_ranks, marker='o', label='Baseline')
plt.plot(keys, main_ranks, marker='o', label='Main Approach')
plt.xlabel('Clothes ID')
plt.ylabel('Rank of Correct Bottom')
plt.title('Comparison of Bottom Ranking (Lower is Better)')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('rank_comparison.png')