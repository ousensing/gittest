import os
import shutil
import subprocess

def create_file(filepath, content):
    with open(filepath, 'w') as file:
        file.write(content)

    print(f'文件已成功创建')

# 指定路径和文件名以及文件内容
# dirpath = os.getcwd()
# filename = 'example.txt'
# filepath = os.path.join(dirpath,filename)
# file_content = '这是文件的内容。'
#
# # 调用函数创建文件
# create_file(filepath, file_content)

# 1. 获取路径下所有文件并写入filelist.sh文件
def get_filenames_and_write_to_filelist(path):
    output = []
    with open('filelist.sh', 'w') as filelist:
        for filename in os.listdir(path):
            if os.path.isfile(filename):
                if not filename.endswith('.sh'):
                    filelist.write(filename + '\n')
                    output.append(os.path.join(path, filename))
    return output


# 2. 遍历复制文件内容至新文件，新文件后缀为.sh
def copy_files_with_sh_extension(filepath_list,protect_list):
    for source_file in filepath_list:
        basename = os.path.splitext(os.path.basename(source_file))[0]  # 获取文件名并去掉原后缀
        dirpath = os.path.dirname(source_file)

        if not source_file.endswith('.sh'):
            with open(source_file, 'r', encoding='utf-8') as source:
                source_content = source.read()
                # 改写为.sh后缀
                target_file = os.path.join(dirpath, basename + '.sh')  # 新文件名
                # 打开目标文件以供写入
                with open(target_file, 'w', encoding='utf-8') as target:
                    # 将源文件的内容写入目标文件
                    target.write(source_content)
            a = os.path.basename(source_file)
            if not os.path.basename(source_file) in protect_list:
                os.remove(source_file)  # 如果新文件名已经存在，删除它


def sh_extension2filelist_type():
    dirpath = os.getcwd()
    basename_list = []
    filename_list = []
    # 读取filelist.sh文件内容并将每一行添加到filelist列表
    with open('filelist.sh', 'r') as filelist_file:
        for line in filelist_file:
            basename = os.path.splitext(os.path.basename(line))[0]  # 获取文件名并去掉原后缀
            basename_list.append(basename)
            filename_list.append(line)


    # 获取当前路径下所有文件
    dirpath = os.getcwd()
    for filename in os.listdir(dirpath):
        if not filename.endswith('.sh'):
            continue
        # 获取文件名并去掉原后缀
        basename = os.path.splitext(os.path.basename(filename))[0]
        if basename in basename_list:
            index = basename_list.index(basename)
            # 使用os.rename()重命名文件
            try:
                if os.path.exists(filename_list[index].strip()):
                    # 如果新文件名已经存在，删除它
                    os.remove(filename_list[index].strip())

                os.rename(filename, filename_list[index].strip())


                print(f"{filename} 已重命名为 {filename_list[index]}")
            except FileNotFoundError:
                print(f"{filename} 不存在，无法重命名")

if __name__ == '__main__':
    # 转成.sh文件
    dirpath = os.getcwd()
    filepath_list = get_filenames_and_write_to_filelist(dirpath)
    protect_file = ['test.py']
    copy_files_with_sh_extension(filepath_list,protect_file)
    # .sh文件转成filelist类型
    # sh_extension2filelist_type()
