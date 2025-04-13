import os

def rename_files(directory, old_str):
    # 遍历指定目录下的所有文件和子目录
    for filename in os.listdir(directory):
        # 拼接完整的文件路径
        file_path = os.path.join(directory, filename)
        # 检查当前路径是否为文件
        if os.path.isfile(file_path):
            # 替换文件名中的指定字符串
            new_filename = filename.replace(old_str, '')
            # 拼接新的文件路径
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed "{file_path}" to "{new_file_path}"')

# 调用函数，传入需要处理的目录和要删除的字符串
rename_files('data/vertebrae/off', '_')
