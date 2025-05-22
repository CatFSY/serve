import os

def rename_prediction_files(base_dir):
    # 遍历 base_dir 目录下的所有文件和文件夹
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # 检查文件名是否以 "_prediction_reinhard.jpg" 结尾
            if file.endswith('_prediction_reinhard.jpg'):
                old_path = os.path.join(root, file)  # 旧文件的完整路径
                new_file_name = file.replace('_prediction_reinhard', '')  # 去掉 "_prediction_reinhard"
                new_path = os.path.join(root, new_file_name)  # 新文件的完整路径
                # 重命名文件
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

def main():
    base_directory = 'AIdjango/dist/UploadvideoProcess/'
    rename_prediction_files(base_directory)


if __name__ == '__main__':
    main()


