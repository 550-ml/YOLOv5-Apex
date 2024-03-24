# -*-coding =utf-8 -*-
# @Time : 2022/12/30 17:24
# @Auther : Wang
# @File :detect.py
# @Software :PyCharm

import math
import threading
import time
import numpy as np
import torch
import pynput
from Sendlnput import *
import win32gui
import win32con
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2,non_max_suppression,  scale_boxes,  xyxy2xywh)
from utils.plots import Annotator
from Screenshoot import screenshot

#  全局变量is_left_pressed,判断鼠标左键是否按下
if_left_pressed = False


#  判断鼠标左键是按下还是松开状态，xy为目标坐标
def mouse_pressed(x, y, button, pressed):
    global if_left_pressed
    if pressed and button == pynput.mouse.Button.left:  # 如果按下，就把这个值变为1
        if_left_pressed = True
    elif not pressed and button == pynput.mouse.Button.left:  # 如果按下，就把这个值变为0
        if_left_pressed = False


#  进入鼠标监听
def mouse_listener():
    with pynput.mouse.Listener(on_click=mouse_pressed) as listener:
        listener.join()


def run():
    global if_left_pressed
    # 加载模型
    # 返回一个torch.device对象，选择的是硬件资源，有CPU,CUDA:0（显卡序号）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 是要加载的网络，weights是权重文件，device是上述设备，fp16是半精度推理，如果是cpu就关上，cuda就true打开
    model = DetectMultiBackend(weights='./weights/yolov5n.pt', device=device, dnn=False, data=False, fp16=False)

    while True:
        # 读取图片
        im = screenshot()  # 调用截图函数进行截图
        im0 = im  # 截图后我们先把这张原图片保存起来
        # 处理图片
        im = letterbox(im, (640, 640), stride=32, auto=True)[0]  # 官方的letterbox函数，可以把图片变为640*640大小
        im = im.transpose((2, 0, 1))[::-1]  # 由我们写的screenshot可知，现在是bgr,我们给变成rgb
        im = np.ascontiguousarray(im)  # 转换成np的数组
        im = torch.from_numpy(im).to(model.device)  # 把im这张图片放到设备上识别（cpu)
        im = im.half() if model.fp16 else im.float()  # 把输入从整型转化为半精度/全精度浮点数
        im /= 255  # 像素点的归一化 从 0 - 255 到 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 添加一个第0维。在pytorch的nn.Module的输入中，第0维是batch的大小，这里添加一个1

        # 推理
        pred = model(im, augment=False, visualize=False)  # 推理结果，pred保存的是所有的bound_box的信息
        # 非极大值抑制  ：也是物体检测算法，依靠分类器对图片中的物体做出矩形框和概率，然后选出最高概率的框，然后遍历其他
        #               框，重叠度超过置信度阈值的就删去一个，重复过程
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000)
        #  以上参数，conf_thres是我上述说的置信度阈值，iou_thres是iou阈值classes是要过滤的类，这里面我们要过滤人，他的编号在
        #  utils\coco128.cal是0，所以这里填0，最后一个max_det是检测框的最大数量

        # 每处理完一张图片，都要进行人物和鼠标移动
        for i, det in enumerate(pred):
                annotator = Annotator(im0, line_width=1)  # 得到一个绘图的类，类中预先存储了原图、线条宽度
                if len(det):
                    distance_list = []  # 用一个列表储存每个人物到中心的距离
                    people_list = []  # 另一个列表存人的信息
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 调整回跟原图一样的大小
                    for *xyxy, conf, cls in reversed(det):  # 处理每个目标的信息
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()  # 将坐标转换成x y w h，并且归一化
                        x = xywh[0] - 320  # 确定人物到屏幕中心的x
                        y = xywh[1] - 320  # 确定人物到屏幕中心的y
                        distance = math.sqrt(x**2 + y**2)  # 根据勾股定理算距离
                        xywh.append(distance)
                        annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{distance}]', color=(34, 139, 34), txt_color=(0, 191, 255))
                        # 绘制边框 分别为对象、标签（这里就是距离）、颜色、文字的颜色
                        distance_list.append(distance)  # 把每个人物的距离加入到列表
                        people_list.append(xywh)  # 同样吧人物的xywh加入到列表
                    people_info = people_list[distance_list.index(min(distance_list))]  # 先找到距离最近的目标，然后再找到最近的对象，
                    # 把这个对象的数据取出来

                    # 如果鼠标按下，就让鼠标移到最小值处
                    if if_left_pressed:
                        mouse_xy(int(people_info[0] - 320), int(people_info[1]) - 320)
                        time.sleep(0.0001)  # 主动睡眠，防止移动过快

                im0 = annotator.result()
                cv2.imshow('win', im0)  # 展示识别的物体
                cv2.waitKey(1)  # 时间暂停
                hwnd = win32gui.FindWindow(None, 'win')  # 第一个参数是类， 第二个是窗口句柄
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                # TOPMOST是置顶的意思，NOMOVE不移动，NOSIZE不改变大小


if __name__ == "__main__":
    threading.Thread(target=mouse_listener).start()
    run()