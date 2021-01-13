# AntiMissile

V0.0.1
-
实现功能：
-
通过二值化实现了对于飞镖的轮廓提取，肉眼看精度尚可，但不知鲁棒性如何，有待测试。

待解决：
-
*param.yml无法正确读取，原因未知。
*误识别了背景光源，这一点准备通过基于混合高斯模型去除背景法来解决，参照网址：[基于混合高斯模型去除背景法](https://blog.csdn.net/weixinhum/article/details/69397787) ，应该能比较快解决。
*双目匹配、计算深度：这一块在检测做的好的前提下后应该很快能够解决，基本就是套函数和公式。

***
V0.0.2
-
实现功能：
-
通过背景减除法实现了目标的检测，完成了深度信息的测算代码。

待解决：
-
*无法应对动态物体
*标定似乎出了问题，两米处有20cm误差

------
V0.0.3
-
实现功能
-
*识别方法从背景减除法改成了基于HSV颜色通道和亮度的识别，达到了和之前相同的进度。
*加入了调整曝光时间的功能
*重新进行了相机的标定，目前进度在三米处为10cm左右，精度显著提升

待解决
-
*实现像素坐标到相机坐标的转换工作，由此计算出俯仰角和偏转角
