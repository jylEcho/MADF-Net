import os
import numpy as np
from scipy.ndimage import distance_transform_edt

# 输入输出路径配置
liver_slices_dir = "/path to your input tumor_slices path/"
liver_dist_dir = "/path to your processed output tumor_slices path/"

# 创建输出目录
os.makedirs(liver_dist_dir, exist_ok=True)


def calc_signed_dist(mask: np.ndarray, target_val: int) -> np.ndarray:
    """计算单个类别的有符号距离图"""
    posmask = (mask == target_val).astype(bool)
    if not posmask.any():
        return np.zeros_like(posmask, dtype=np.float32)

    negmask = ~posmask
    pos_dist = distance_transform_edt(posmask)  # 内部距离
    neg_dist = distance_transform_edt(negmask)  # 外部距离

    # 符号规则：前景内部为负，背景外部为正，边界为0
    signed_dist = neg_dist * negmask - (pos_dist - 1) * posmask
    return signed_dist.astype(np.float32)


# 处理肝脏切片
for filename in os.listdir(liver_slices_dir):
    if not filename.endswith(".npy"):
        continue

    # 读取肝脏二值图
    liver_mask_path = os.path.join(liver_slices_dir, filename)
    liver_mask = np.load(liver_mask_path)  # 假设肝脏二值图值域为0和1

    # 计算肝脏距离图（目标值1）
    liver_dist = calc_signed_dist(liver_mask, 1)
    np.save(os.path.join(liver_dist_dir, filename), liver_dist)

    print(f"Processed liver slice: {filename}")
