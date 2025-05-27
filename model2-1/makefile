# Makefile

# 你可以在命令行用 `make all`、`make extract`、`make train`、`make eval`、`make infer`

PYTHON = python3
TOP_IMAGE = ./Cleaned-Maryland-Dataset/100031484/top.jpg

.PHONY: all extract train eval infer clean

all: extract train eval

# 1. 抽特徵
extract:
	@echo "==> Extracting Model1 features..."
	$(PYTHON) extract_feats.py \
	  --data_root ./Cleaned-Maryland-Dataset \
	  --model1_ckpt ./label_train/saved_models/best_tagger.pth \
	  --output ./train_feats.pt

# 2. 訓練 Model2
train:
	@echo "==> Training CompatibilityNet (Model2)..."
	$(PYTHON) train_compat.py

# 3. 評估
eval:
	@echo "==> Evaluating on validation set..."
	$(PYTHON) eval.py

# 4. 推論示例
infer:
	@echo "==> Running inference on sample top image..."
	$(PYTHON) infer.py --top_image $(TOP_IMAGE)

# 清理（可选）
clean:
	rm -f train_feats.pt bot_feats_cache.pt
	rm -rf checkpoints/*.pth
	rm -f *loss_curve.png
