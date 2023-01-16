# -*-coding =utf-8 -*-
# @Time : 2022/12/29 11:08
# @Auther : Wang
# @File :Screenshoot.py
# @Software :PyCharm
import cv2
import mss
import numpy
with mss.mss() as m:
    pass

# 定义截图范围，X_long,Y_long是你的笔记本的分辨率，游戏本一般是1920*1080
X_long = 1920
Y_long = 1080
picture_size = (int(X_long / 2 - 320), int(Y_long / 2 - 320), int(X_long / 2 + 320), int(Y_long / 2 + 320))
# 截图是需要左上角和右下角的坐标的
Screenshot_value = m


# 截图函数
def screenshot():
    img = Screenshot_value.grab(picture_size)  # 调用mss.grab()方法进行截图，给出坐标参数
    img = numpy.array(img)  # 截完图把数据编程np数组,你输出img就会发现他是个四通道的
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 四通道转三通道
    return img
