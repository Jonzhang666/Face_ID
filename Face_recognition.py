import cv2 as cv
import sys
import gc
from faces_train import Model

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model('./model/Jonzhang.face.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 0, 255)

    # 捕获指定摄像头的实时视频流
    cap = cv.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "E:/python-3.6.6-64-bit/opencv-master/data/haarcascades/haarcascade_frontalface_alt_tree.xml"

    # 循环检测识别人脸
    while True:
        ref, frame = cap.read()  # 读取一帧视频
        if not ref:
            break

        # 加入该段代码将使拍出来的画面呈现镜像效果
        # 第二个参数为视频是否上下颠倒 0为上下颠倒 1为不进行上下颠倒
        frame = cv.flip(frame, 1)
        # 图像灰化，降低计算复杂度
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 使用人脸识别分类器，读入分类器
        cascade = cv.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        # 限定头像尺寸
        h_1 = 120
        w_1 = 120

        if len(faceRects) == 1:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                if abs(w_1 - w) > 20 and abs(h_1 - h) > 20:
                    font = cv.FONT_HERSHEY_SIMPLEX  # 字体类型 1：字号，4：线形
                    cv.putText(frame, 'Adjust Distance', (30, 30), font, 0.5, (0, 0, 255), 2)
                    break
                w_1 = w
                h_1 = h

                # 截取脸部图像提交给模型识别这是谁
                image = frame[y: y + h, x: x + w]
                faceID = model.face_predict(image)

                cv.rectangle(frame, (x, y), (x + w, y + h), color, thickness=1)
                # 如果是“我”
                if faceID == 0:
                    # 文字提示是谁
                    cv.putText(frame, 'Jonzhang',
                                (30, 30),  # 坐标
                                cv.FONT_HERSHEY_SIMPLEX,  # 字体
                                0.5,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                else:
                    pass

        cv.imshow("Face_recognition", frame)

        # 等待10毫秒看是否有按键输入
        k = cv.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv.destroyAllWindows()