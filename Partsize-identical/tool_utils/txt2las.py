import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import laspy
    import numpy as np
    import os
    return laspy, mo, np, os


@app.cell
def __(laspy, np, os):


    def load_data(path,name):
        
        # 读取数据
        data = np.loadtxt(f'{path}\\log\\sem_seg\\2024-10-05_01-43\\visual\\Bridge4_1_pred.txt', delimiter=' ')
        
        girder_data = data[data[:, 6] == 1]
        girder_coords = girder_data[:, :2]
        
        # 分离各列数据
        x, y, z, r, g, b, label =  girder_data.T
        
        r_16bit = (r * 65535).astype(np.uint16)
        g_16bit = (g * 65535).astype(np.uint16)
        b_16bit = (b * 65535).astype(np.uint16)
        
        # 创建LAS文件
        las = laspy.create(file_version="1.3", point_format=3)
        
        # 设置头部信息
        las.header.offsets = [np.min(x), np.min(y), np.min(z)]
        las.header.scales = [0.001, 0.001, 0.001]
        
        # 写入点数据
        las.x = x
        las.y = y
        las.z = z
        las.red = r_16bit
        las.green = g_16bit
        las.blue = b_16bit
        las.classification  = label.astype(np.uint8)
        
        name = 'bridge-4-1'
        
        # 保存LAS文件
        las.write(name+'.las')
        
        print(f"LAS文件已创建：{name}.las")



    path=os.getcwd()
    return load_data, path


@app.cell
def __():
    1
    return


if __name__ == "__main__":
    app.run()
