import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 假设这是你的相似性矩阵，它是一个二维数组
for i in range(1):
    similarity_matrix = torch.load('/public/ly/SBF/sim_matrix_iter8_lora_batch64.pth'.format(i))
    print(similarity_matrix[:3,:3])

    plt.figure(figsize=(8, 6))  # 可以调整图形大小
    sns.heatmap(similarity_matrix, annot=False, fmt='.2f', cmap='coolwarm', cbar=True, vmax=1, vmin=0)

    # 添加标题和标签
    # plt.title('Oneshot')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig('/public/ly/SBF/img/sim_matrix_llama3.1_iter8_batch64.pdf')

# 显示图形
# plt.show()

# from PIL import Image
# import glob
#
# def create_gif(image_folder, output_path, duration=500):
#     # 获取图片文件路径列表
#     image_files = glob.glob(f"{image_folder}/*.png")  # 假设图片是 PNG 格式
#     image_files.sort()  # 确保图片按照文件名顺序排列
#
#     # 打开图片并将其添加到列表
#     images = [Image.open(img) for img in image_files]
#
#     # 保存为 GIF
#     images[0].save(
#         output_path,
#         save_all=True,
#         append_images=images[1:],  # 后续图片
#         duration=duration,  # 每帧显示的时间（毫秒）
#         loop=0  # 无限循环
#     )
#
# # 使用示例
# image_folder = '/public/ly/SBF/img'
# output_path = 'llama3.1_oneshot_animation.gif'
# create_gif(image_folder, output_path)
