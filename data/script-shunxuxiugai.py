import os
import glob
from PIL import Image
import shutil

# 图片所在的文件夹路径
image_folder = 'F:\\ffhq'  # 替换为你的图片文件夹路径
# 输出文件夹路径
output_folder = 'F:\\ffhq2'  # 替换为你想要保存新命名图片的文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化计数器
counter = 1

# 遍历图片文件夹中的所有图片文件
for image_file in glob.glob(f"{image_folder}/*.png"):
    # 获取图片的原始名称
    original_name = os.path.basename(image_file)
    # 创建新的文件名
    new_name = f"{output_folder}/{counter:05d}.png"
    # 复制图片到新的位置，并使用新的文件名
    shutil.copy2(image_file, new_name)
    # 更新计数器
    counter += 1

print(f"Renamed {counter - 1} images.")