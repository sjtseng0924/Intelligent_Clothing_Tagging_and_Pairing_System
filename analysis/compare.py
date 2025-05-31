baseline_path = "rank_baseline.txt"
main_path = "rank_main.txt"

# 讀取 baseline
baseline_dict = {}
with open(baseline_path) as f:
    for line in f:
        k, v = line.strip().split()
        baseline_dict[k] = int(v)

# 讀取 main
main_dict = {}
with open(main_path) as f:
    for line in f:
        k, v = line.strip().split()
        main_dict[k] = int(v)

# 比較
main_better = 0
baseline_better = 0
equal = 0
for k in baseline_dict:
    if k in main_dict:
        if main_dict[k] < baseline_dict[k]:
            main_better += 1
        elif main_dict[k] > baseline_dict[k]:
            baseline_better += 1
        else:
            equal += 1

print(f"main < baseline: {main_better}")
print(f"main > baseline: {baseline_better}")
print(f"main == baseline: {equal}")