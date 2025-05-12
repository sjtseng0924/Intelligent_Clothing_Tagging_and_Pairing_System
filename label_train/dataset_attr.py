import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DeepFashionAttrDataset(Dataset):
    def __init__(self, img_root, attr_file, eval_file, mode='train', transform=None, max_items=None):
        self.img_root = img_root
        self.transform = transform

        # 讀取切分
        eval_df = pd.read_csv(eval_file, sep='\s+', skiprows=2, header=None, names=["image_path", "split"])
        eval_df = eval_df[eval_df["split"] == mode]

        # 讀取屬性標籤
        attr_df = pd.read_csv(attr_file, sep='\s+', skiprows=2, header=None)
        attr_df.iloc[:, 1:] = attr_df.iloc[:, 1:].replace(-1, 0)  # 將 -1 轉為 0
        attr_df.columns = ['image_path'] + [f'attr_{i}' for i in range(attr_df.shape[1] - 1)]

        # 合併
        df = pd.merge(eval_df, attr_df, on='image_path')

        if max_items:
            df = df.iloc[:max_items]

        self.image_paths = [os.path.join(img_root, p) for p in df['image_path']]
        self.labels = df.iloc[:, 2:].astype('float32').values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
