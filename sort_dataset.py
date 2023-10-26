import pandas as pd
import os
import shutil

"""从12611个训练集中获取1000张图片，并对少于10张的标签增强到10张，最终能够获得2000+图片"""


def copyfile(fname, target_dir):
    """将文件复制到指定文件夹"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(fname, target_dir)

def sortDataset(data_dir, DF):
    """创建新的数据集"""
    length = len(DF)
    for idx, row in DF.iterrows():
        # 获取当前文件路径
        filename = os.path.join(data_dir, 'boneage-training-dataset', f"{row['id']}.png")
        # 获取该文件的标签
        label = str(row['boneage'])
        copyfile(filename, os.path.join('../../autodl-tmp', 'train', label))

        if int(idx) > length/2:
            break
            


def reorg_aug_data(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'boneage-training-dataset.csv'))
    sortDataset(data_dir, df)

if __name__ == '__main__':
    data_dir = '../archive'
    reorg_aug_data(data_dir=data_dir)