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
def __(np):
    def load_data(path,part):
        
        # 读取数据
        data = np.loadtxt(path, delimiter=' ')

        part_data = data[data[:, 6] == part]
        deck_coords = part_data[:, :part]
        
        
    return (load_data,)


@app.cell
def __(__file__, os):
    def main():
        # 获取当前脚本的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 更改当前工作目录
        os.chdir(script_dir)

        path=os.getcwd()

        pathraw="..\\data\\bridge-5cls-fukushima\\test\\Bridge4_1.txt"
        pathtest=f'{path}\\log\\sem_seg\\2024-10-05_01-43\\visual\\Bridge4_1_pred.txt'

        #{0: 'abutment', 1: 'girder', 2: 'deck', 3: 'parapet', 4: 'noise'}
        
    return (main,)


if __name__ == "__main__":
    app.run()
