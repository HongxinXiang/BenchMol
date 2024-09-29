import numpy as np
import torch
from PIL import Image
import cv2


def get_labels_from_df(df, task_type, y_column="label"):
    """examples of label: 1 0 1 -1 0"""
    if task_type == "classification":
        labels = np.array(df[y_column].apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())
    elif task_type == "regression":
        labels = np.array(df[y_column].apply(lambda x: np.array(str(x).split(" ")).astype(float).tolist()).tolist())
    else:
        raise UserWarning("{} is undefined.".format(task_type))
    return  labels


def wrap_with_max_len(data, max_len):
    shape = np.array(data.shape)
    shape[0] = max_len
    new_data = torch.zeros(*shape, dtype=data.dtype)
    new_data[:data.shape[0]] = data
    return new_data


def read_image(image_path, img_type="RGB"):
    """从 image_path 从读取图片
    如果 img_type="RGB"，则直接读取；
    如果 img_type="BGR"，则将 BGR 转换为 RGB
    """
    if img_type == "RGB":
        return Image.open(image_path).convert('RGB')
    elif img_type == "BGR":
        img = Image.open(image_path).convert('RGB')  # iamge_path 中的图片是以 BGR 的格式存储的
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
        img = Image.fromarray(img)
        return img
    else:
        raise NotImplementedError