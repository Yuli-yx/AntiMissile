# AntiMissile

V0.0.1
-
实现功能：
-
通过二值化实现了对于飞镖的轮廓提取，肉眼看精度尚可，但不知鲁棒性如何，有待测试。

待解决：
-
1. param.yml无法正确读取，原因未知。
2. 误识别了背景光源，这一点准备通过基于混合高斯模型去除背景法来解决，参照网址：[基于混合高斯模型去除背景法](https://blog.csdn.net/weixinhum/article/details/69397787) ，应该能比较快解决。
3. 双目匹配、计算深度：这一块在检测做的好的前提下后应该很快能够解决，基本就是套函数和公式。

***
