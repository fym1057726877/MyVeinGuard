import torch
import os
from matplotlib import pyplot as plt


def get_project_path(project_name='MyVeinGuard'):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name



if __name__ == '__main__':
    print(get_project_path())