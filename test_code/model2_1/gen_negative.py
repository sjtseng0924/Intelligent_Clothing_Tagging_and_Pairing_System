# gen_negative_pairs.py
import os, random, csv

root_dir = "Cleaned-Maryland-Dataset"   # 改成你的路徑
ids = sorted(d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)))

random.seed(42)
neg_pairs = []
# 每個正例 top_id，隨機選一個不同的 bot_id 當負例
for top_id in ids:
    bot_id = random.choice([i for i in ids if i != top_id])
    neg_pairs.append((top_id, bot_id))

# 寫入 CSV
with open("negative_pairs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["top_id", "bot_id"])
    writer.writerows(neg_pairs)

print(f"已產生 {len(neg_pairs)} 對負例，存於 negative_pairs.csv")
