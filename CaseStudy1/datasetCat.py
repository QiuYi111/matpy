from datasets import load_from_disk, concatenate_datasets

# 1. 从本地加载两个数据集
dataset1 = load_from_disk("/Volumes/DataHub/dataProcessed/12/datasetSVD")
dataset2 = load_from_disk("/Volumes/DataHub/dataProcessed/bubu/datasetSVD")

# 2. 验证特征结构是否一致（关键步骤）
assert dataset1.features == dataset2.features, "数据集结构不一致"

# 3. 使用concatenate_datasets合并
combined_dataset = concatenate_datasets([dataset1, dataset2])

# 4. 保存合并后的数据集
combined_dataset.save_to_disk("/Volumes/DataHub/dataProcessed/12-bubu")
