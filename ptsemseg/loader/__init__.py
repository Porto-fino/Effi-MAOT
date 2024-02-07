import json

from ptsemseg.loader.airsim_loader import airsimLoader


def get_loader(name): #该函数为ptsemseg项目简化的loader，函数的意思是，根据“name”的内容，返回不同加载器,这里只会返回airsimLoader
    """get_loader

    :param name:
    """
    return {
        "airsim": airsimLoader,
    }[name]

