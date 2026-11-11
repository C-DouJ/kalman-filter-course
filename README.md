# 从零学卡尔曼滤波

一门可以动手拖的卡尔曼滤波入门课：每个概念都有交互图、逐步推导，以及 Python 和 MATLAB 代码。

**在线阅读**：https://c-douj.github.io/kalman-filter-course/

## 目录

主干
1. 为什么要滤波
2. 权重该怎么选（线性最小方差，不需要高斯假设）
3. 单步跑一遍（预测与更新、初值的影响）

选读 · 换个角度看
4. 贝叶斯视角：高斯给了我们什么
5. 极大后验视角：加权最小二乘与两根弹簧
6. 几何视角（即将上线）
7. 工程视角与十视角总览（即将上线）

入门篇只讲标量、时不变的例子；进阶篇会推广到一般的线性时变系统。

## 文件

- `index.html`：课程网页，双击即可离线打开
- `code/`：每一课的 Python 和 MATLAB 代码

## 运行代码

Python 需要 `numpy` 和 `matplotlib`：

```
pip install numpy matplotlib
python code/lesson3_kalman_1d.py
```

MATLAB 直接运行对应的 `.m` 文件（R2018b 及以上）。

## 反馈

发现错误或有建议，欢迎提 Issue。
