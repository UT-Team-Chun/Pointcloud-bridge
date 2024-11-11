# read las file
import laspy
import numpy as np


def read_las_file(las_path):
    """
    读取las文件并返回与原格式相同的数据 (N×7的数组，包含xyzrgb和label)
    """
    # 读取las文件
    las = laspy.read(las_path)

    # 获取xyz坐标
    x = las.x
    y = las.y
    z = las.z

    # 获取RGB值 (las文件中通常RGB值范围是0-65535，需要转换到0-255)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = las.red / 65535 * 255
        g = las.green / 65535 * 255
        b = las.blue / 65535 * 255
    else:
        # 如果没有颜色信息，设置默认值
        r = np.zeros_like(x)
        g = np.zeros_like(x)
        b = np.zeros_like(x)

    # 获取分类标签 (如果存在)
    if hasattr(las, 'classification'):
        labels = las.classification
    else:
        # 如果没有标签，设置默认值0
        labels = np.zeros_like(x)

    # 组合所有数据
    points = np.column_stack((x, y, z, r, g, b))
    bridge_data = np.column_stack((points, labels))

    return bridge_data