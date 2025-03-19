from PIL import Image
import torch
from torchvision.transforms import v2
from datasets import Dataset, Features, Array3D,Value
import random


def generate_variants(image_path, n=5, target_size=(256, 256),to_mat=False):
    """
    生成N张灰度化、随机变换的变体，输出展平的NumPy数组
    :param image_path: 原始图片路径
    :param n: 生成变体数量
    :param target_size: 统一输出尺寸
    :return: 包含展平数组的数据集
    """
    # 加载原始图片并统一尺寸
    img = Image.open(image_path).convert('RGB')
    resize = v2.Resize(target_size)
    img = resize(img)
    
    # 定义变换组合（强制灰度化+随机参数）
    transform = v2.Compose([
        #v2.RandomRotation(degrees=random.choice([0, 90, 180, 270])),
        #v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(
            brightness=random.uniform(0.5, 1.5),
            contrast=random.uniform(0.5, 1.5)
        ),
        v2.Grayscale(num_output_channels=1),  # 强制灰度化
    ])
    
    # 生成变体
    variants = []
    labels = []
    for _ in range(n):
        transformed_img = transform(img)  # 输出形状: (H*W, )
        variants.append(transformed_img)
        labels.append(1)  # 假设标签为0
        
    # 构建数据集
    dataset = Dataset.from_dict(
        {"image": variants, "label": labels})
    return dataset

# 使用示例
if __name__ == "__main__":
    dataset = generate_variants("/Users/jingyi/Desktop/12.jpg", n=5000, target_size=(100, 100))
    dataset.save_to_disk("/Volumes/DataHub/dataProcessed/12")
    img=dataset["image"][0]
    img.show()
