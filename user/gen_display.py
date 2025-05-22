import cv2
from django.http import StreamingHttpResponse
import atexit
import os
import cv2
from datetime import datetime
import sys
import time
from django.http import JsonResponse
from django.shortcuts import render
import os
from django.conf import settings
from django.conf import settings
import os
import numpy as np
from PIL import Image
import cv2
import urllib.parse
import mimetypes
import re
import threading
import json
from django.conf import settings
from django.shortcuts import render
import subprocess
from django.core.cache import cache
import io
import paddlex as pdx
import os
import cv2

image_path = None
# 加载模型并预测
BigModel = pdx.load_model('output/hrnet/best_model')
SmallModel = pdx.load_model('output/hrnet/best_model')



sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
current_directory = os.getcwd()
module_directory = os.path.join(current_directory)
from django.http import StreamingHttpResponse
from wsgiref.util import FileWrapper
from django.http import HttpResponse
from multiprocessing import Process, Manager, Event


params = None
isrecord = None
starttime = None
endtime = None
camera = None
queueISdeal = False
camId = 0
background_thread = None
save_thread =None
RecordCounter = None
from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
def getAllCam(request):
    # cameras = list_video_devices()
    cameras = [(0, 'HD Webcam'), (1, 'wifi Camera')]
    print(cameras)
    return JsonResponse({'cam': cameras , "success": 1}, status=200)
def Camchoice(request):
    global camId
    global camera
    data = json.loads(request.body)
    print(data)
    if data is not None:
        camId = data.get("camId")
        if camera is not None:
            camera.release()  # 释放摄像头资源
            cv2.destroyAllWindows()
            camera = None  # 清空摄像头对象
        return JsonResponse({ "success": 1}, status=200)
    else:
        return JsonResponse({ "success": 0}, status=200)

def Sharpening(img, method='unsharp', alpha=1.5, kernel_size=(5, 5)):
    """
    对输入图像进行锐化处理

    Args:
        img (np.ndarray): 输入图像（BGR格式，H×W×3）
        method (str): 'unsharp' 或 'laplacian'
        alpha (float): 锐化强度系数（unsharp时有效）
        kernel_size (tuple): 高斯模糊核大小（unsharp时使用）

    Returns:
        np.ndarray: 锐化后的图像（BGR）
    """
    if method == 'unsharp':
        # 1. 高斯模糊
        blurred = cv2.GaussianBlur(img, kernel_size, 0)
        # 2. 原图 + 残差（锐化）
        sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
        return sharpened

    elif method == 'laplacian':
        # 1. 灰度图用于边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.clip(lap, 0, 255).astype(np.uint8)
        lap_color = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(img, 1.0, lap_color, 0.7, 0)
        return sharpened

    else:
        raise ValueError("method must be 'unsharp' or 'laplacian'")
    

def fastElimination(img, blur_ksize=11, threshold=20, debug=False):
    """
    基于局部对比度的缺陷检测（适合快速剔除无缺陷图像）

    参数:
        img: BGR 格式图像 (H, W, 3)
        blur_ksize: 高斯模糊核大小，值越大，越模糊
        threshold: 差值图像的均值阈值，超过说明存在突变区域
        debug: 是否显示中间图像（用于调试）

    返回:
        has_defect: 是否包含缺陷 (True/False)
        diff_map: 差值图（可选调试用）
    """
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊处理
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 计算原图与模糊图之间的差值图（突变区域）
    diff = cv2.absdiff(gray, blurred)

    # 可选调试：显示差值图
    if debug:
        cv2.imshow("Difference", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 计算差值图的全图均值
    diff_mean = np.mean(diff)

    # 判断：差值均值过大，说明有突变区域，可能是缺陷
    has_defect = diff_mean > threshold

    return has_defect, diff

def CLAHE(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    使用 CLAHE 对图像进行局部对比度增强（适用于BGR格式图像）

    Args:
        img (np.ndarray): 输入图像（BGR格式，H×W×3）
        clip_limit (float): 对比度限制阈值，防止过度放大噪声（默认2.0）
        tile_grid_size (tuple): 分块大小（默认为8x8）

    Returns:
        np.ndarray: 增强后的图像（BGR）
    """
    # 转换为 LAB 色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建 CLAHE 对象并应用到 L 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # 合并通道再转回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return img_clahe
def detail(img, sigma=1.5, weight=1.5):
    """
    使用高通滤波增强图像细节

    Args:
        img (np.ndarray): 输入图像（BGR格式，H×W×3）
        sigma (float): 高斯模糊的标准差（控制模糊程度，默认1.5）
        weight (float): 高频成分增强的权重（默认1.5）

    Returns:
        np.ndarray: 增强后的图像（BGR）
    """
    # 对原图进行高斯模糊，得到低频信息
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # 提取高频成分，即细节部分
    detail = cv2.subtract(img, blurred)

    # 对细节部分加权增强
    enhanced = cv2.addWeighted(img, 1, detail, weight, 0)

    return enhanced


def deblurring(img, kernel_size=5, iterations=10):
    """
    使用反卷积去模糊图像

    Args:
        img (np.ndarray): 输入图像（BGR格式，H×W×3）
        kernel_size (int): 模糊核大小，决定去模糊时的滤波范围，越大去模糊效果越强（默认5）
        iterations (int): 迭代次数，越大去模糊效果越明显（默认10）

    Returns:
        np.ndarray: 去模糊后的图像（BGR）
    """
    # 创建一个模糊核（假设是均匀模糊的）
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # 通过反卷积进行去模糊处理
    result = img.copy()
    for _ in range(iterations):
        result = cv2.filter2D(result, -1, kernel)

    # 可选：可以进一步增强图像的清晰度（锐化）
    result = cv2.addWeighted(result, 1.5, img, -0.5, 0)
    
    return result
def determine_process_type(params):
    # 创建一个空列表，用于保存所有值为True的处理类型
    active_process_types = []
    
    # 遍历params字典中的所有键值对，找到值为True的键
    for key, value in params.items():
        if value:  # 如果该处理类型为True
            active_process_types.append(key)  # 将该处理类型添加到列表中
    
    # 如果有任何处理类型为True，返回这些处理类型的列表；否则，返回默认值 "other"
    if active_process_types:
        return active_process_types
    else:
        return ["small"]  # 如果没有处理类型被启用，返回默认值 ["other"]

def initialize():
    global params
    global isrecord
    global RecordCounter
    global background_thread
    try:
        
        if params is None:
            params = {
            'size': False,
            "detail":False,#细节增强
            "fastElimination":False,#快速剔除
            "dark":False,#去模糊
            "CLAHE":False,#局部对比度增强
            "Sharpening":False #图像锐化
        }
        if isrecord is None:
            isrecord = False
        if RecordCounter is None:
            RecordCounter=0
        background_thread = threading.Thread(target=background_processing, daemon=True)
        background_thread.start()
        if not os.path.exists('AIdjango/dist/videotemp/'):
            os.makedirs('AIdjango/dist/videotemp/')
        if not os.path.exists('AIdjango/dist/livedisplay/'):
            os.makedirs('AIdjango/dist/livedisplay/')
        if not os.path.exists('AIdjango/dist/livedisplay/people'):
            os.makedirs('AIdjango/dist/livedisplay/people')
        if not os.path.exists('AIdjango/dist/livedisplay/vehicle'):
            os.makedirs('AIdjango/dist/livedisplay/vehicle')
        if not os.path.exists('AIdjango/dist/livedisplay_record/'):
            os.makedirs('AIdjango/dist/livedisplay_record/')
        if not os.path.exists('AIdjango/dist/livedisplay_record2video/'):
            os.makedirs('AIdjango/dist/livedisplay_record2video/')
        if not os.path.exists('AIdjango/dist/UploadvideoProcess/'):
            os.makedirs('AIdjango/dist/UploadvideoProcess/')             
        if not os.path.exists('AIdjango/dist/UploadphotoSave/'):
            os.makedirs('AIdjango/dist/UploadphotoSave/')                 
        if not os.path.exists('AIdjango/dist/UploadvideoSave/'):
            os.makedirs('AIdjango/dist/UploadvideoSave/')  
        if not os.path.exists('AIdjango/dist/UploadphotoProcess/'):
            os.makedirs('AIdjango/dist/UploadphotoProcess/')     
        if not os.path.exists('AIdjango/dist/livedisplay_recordphoto/'):
            os.makedirs('AIdjango/dist/livedisplay_recordphoto/')                 
    except Exception as e:
        print(f"Error initializing models: {e}")
        return HttpResponse("Error initializing models.", status=500)

    return HttpResponse("Models initialized and ready.")
def index(request):
    return render(request,"index.html")
def ConfirmParams(request):
    global params
    data = json.loads(request.body)
    params = {
            "size":data.get('size'),#快模型
            "detail":data.get('detail'),#细节增强
            "fastElimination":data.get('fastElimination'),#快速剔除
            "dark":data.get('dark'),#去模糊
            "CLAHE":data.get('CLAHE'),#局部对比度增强
            "Sharpening":data.get('Sharpening') #图像锐化
    }
    return JsonResponse({'message': "success parms", "success": 1}, status=200)

def get_camera_frame_size(camera):
    ret, frame = camera.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        return frame.shape[1], frame.shape[0]  # 返回宽度和高度
    return None


def background_processing():
    pass

import requests
def open_camera(request):
    """
    关闭摄像头路由。
    """
    global camera
    global queueISdeal
    queueISdeal = False
    if camera is  None:
        if camId==1:
            stream_url = "http://192.168.213.160/"
            requests.get("http://192.168.213.160/mode?val=1")
            camera = cv2.VideoCapture(stream_url)
        else:
            camera = cv2.VideoCapture(camId)  
    return JsonResponse({'status': 'Camera open'})
def close_camera(request):
    """
    关闭摄像头路由。
    """
    global camera
    global queueISdeal
    queueISdeal =  False
    if camera is not None:
        camera.release()  # 释放摄像头资源
        cv2.destroyAllWindows()
        camera = None  # 清空摄像头对象
    return JsonResponse({'status': 'Camera closed'})

# def changestyle():
#     global paddledetection_net
#     global queueISdeal
#     # paddledetection_net.vehicle_waitting_dealwith_queue=[]
#     # paddledetection_net.people_waitting_dealwith_queue=[]
#     # paddledetection_net.newStart()
#     clear_directory('AIdjango/dist/livedisplay/vehicle')
#     clear_directory('AIdjango/dist/livedisplay/people')
#     queueISdeal  = True
    
t_start = time.time()
t_end = time.time()
def gen_display(camera):
    global RecordCounter
    global queueISdeal 
    
    target_size = get_camera_frame_size(camera)
    RecordCounter=0
    target_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record")
    items = os.listdir(target_dir)
    folders = [item for item in items if os.path.isdir(os.path.join(target_dir, item))]
    folder_count = len(folders)
    last_request_time=time.time()
    while True:
        # 读取图片
        if camera is None:
            break
        camId==1
        if camId == 1:
            now = time.time()
            if now - last_request_time < 0.5:
                continue
            last_request_time = now
        ret, frame = camera.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)#BGR格式
            # 将图片进行解码                
            if ret:
                img = frame
                try:
                    defect_info={}
                    if params["dark"]:
                        print(2)
                    if params["CLAHE"]:
                        img = CLAHE(img)
                    if params["detail"]:
                        img = detail(img)
                    if params["Sharpening"]:
                        img = Sharpening(img)
                    if params["size"]:
                        result= BigModel.predict(img)
                    else:
                        result = SmallModel.predict(img)
                        label_map = result['label_map']
                        unique_labels = np.unique(label_map)  # 获取所有唯一标签值
                        num_defects = len(unique_labels) - 1  # 减去背景类别
                        defect_info["num_defects"] = num_defects
                        defect_info["detected_labels"] = unique_labels.tolist()  # 将标签转换为列表以便保存
                        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        save_path = f'AIdjango/dist/livedisplay_recordphoto/{current_time+".jpg"}'
                        if num_defects:
                            vis_result = pdx.seg.my_visualize(frame, result, weight=0.4, save_dir=save_path,color=[0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255])
                            photo_instance = Photo(
                            photo_name=current_time+".jpg",  # 照片名称
                            photo_path=save_path,  # 原始照片路径
                            source = "live",
                            result_path=save_path,  # 处理后的照片路径
                            process_type=determine_process_type(params),  # 处理类型（你可以在params中加入多个处理类型）
                            upload_time=timezone.now(),  # 上传时间（假设这里使用当前时间）
                            process_time=timezone.now(),  # 处理时间（假设这里使用当前时间）
                            defect_info=defect_info  # 缺陷信息（包括快速剔除和其他处理类型）
                        )

                            photo_instance.save()
                        else:
                            vis_result = frame
                    if isrecord:
                        if RecordCounter==0:
                                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                save_dir = f'AIdjango/dist/livedisplay_record/{current_time}'
                                os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{RecordCounter}.jpg")
                        # print(save_path)
                        cv2.imwrite(save_path, vis_result)  # 保存为BGR格式
                        RecordCounter += 1
                    # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    frame = vis_result
                    t_end = time.time()
                    t = t_end - t_start
                    if t == 0:
                        t = 1
                    ret, frame = cv2.imencode('.jpeg', frame)
                    # 递增计数器
                    
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
                except:
                    pass

def list_files_with_sizes(folder_path):
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print("文件夹不存在！")
        return

    total_size = 0  # 初始化总大小变量

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # 确保是文件而不是文件夹
            file_size = os.path.getsize(file_path)  # 获取文件大小（字节）
            total_size += file_size  # 累加文件大小
            print(f"{filename}: {file_size / (1024 * 1024):.2f} MB")  # 转换为MB并打印

    # 打印总大小并返回（以MB为单位）
    print(f"总大小: {total_size / (1024 * 1024):.2f} MB")
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    return total_size_mb   # 返回总大小（MB）
def delete(request):
    data = json.loads(request.body)
    type = data.get("type")
    print(type)
    if type=="record":

        clear_directory('AIdjango\dist\livedisplay_record2video')
        return JsonResponse({ "success": 1}, status=200)
    elif type=="video":
        clear_directory('AIdjango/dist/UploadvideoProcess/')
        return JsonResponse({ "success": 1}, status=200)
    elif type=="photo":
        clear_directory('AIdjango/dist/UploadphotoProcess/')
        return JsonResponse({ "success": 1}, status=200)
    else:
        return JsonResponse({ "message":"None"}, status=200)
def get_sizes(request):
    data = json.loads(request.body)
    type = data.get("type")
    print(type)
    if type=="record":
        return JsonResponse({ "size": list_files_with_sizes("AIdjango/dist/livedisplay_record2video/")}, status=200)
    elif type=="video":
        return JsonResponse({ "size": list_files_with_sizes("AIdjango/dist/UploadvideoProcess/")}, status=200)
    elif type=="photo":
        return JsonResponse({ "size": list_files_with_sizes("AIdjango/dist/UploadphotoProcess/")}, status=200)
    else:
        return JsonResponse({ "message":"None"}, status=200)

def video_record_on(request):
    global isrecord
    global RecordCounter 
    global camera
    global starttime
  
    if request.method == 'POST':
        starttime = time.time()
        if camera is  None:
            camera = cv2.VideoCapture(camId)  
        isrecord = True
        RecordCounter = 0
        return JsonResponse({'status': 'start record'})
    return JsonResponse({'status': 200})
def video_record_off(request):
    global RecordCounter
    global isrecord
    global endtime
    if request.method == 'POST':
        isrecord = False
        endtime = time.time()
        RecordCounter = 0
        save_thread = threading.Thread(target=saverecord)
        save_thread.start()
        return JsonResponse({'status': 'process finish'})
    return JsonResponse({'status': ""})

# def saverecord():
#         base_dir = 'AIdjango/dist/livedisplay_record'

#         folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

#         # 按时间戳排序，找到最新的文件夹
#         if folders:
#             latest_folder = max(folders, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))
#             save_photo_dir = os.path.join(base_dir, latest_folder)
#         image_files = [f for f in os.listdir(save_photo_dir) if f.endswith('.jpg') or f.endswith('.png')]
#         image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

#         # 检查是否有图像
#         if not image_files:
#             print("没有找到任何图像文件。")
#             exit()

#         # 获取第一张图像以获取宽高
#         first_image_path = os.path.join(save_photo_dir, image_files[0])
#         frame = cv2.imread(first_image_path)
#         height, width, layers = frame.shape

#         # 定义视频编写器
#         save_video_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record2video")
#         video_name = os.path.join(save_video_dir, latest_folder+'.avi')
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码方式
#         video_writer = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))  # 30 FPS

#         # 读取并写入图像到视频
#         for image_file in image_files:
#             image_path = os.path.join(save_photo_dir, image_file)
#             frame = cv2.imread(image_path)
#             video_writer.write(frame)

#         # 释放视频编写器
            
#         video_writer.release()


def saverecord():
    global starttime
    global endtime
    base_dir = 'AIdjango/dist/livedisplay_record'

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # 按时间戳排序，找到最新的文件夹
    if folders:
        latest_folder = max(folders, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))
        save_photo_dir = os.path.join(base_dir, latest_folder)
    else:
        print("没有找到任何文件夹。")
        return

    image_files = [f for f in os.listdir(save_photo_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # 检查是否有图像
    if not image_files:
        print("没有找到任何图像文件。")
        return
    recording_duration = endtime - starttime  # 以秒为单位

    # 计算帧率
    frame_count = len(image_files)
    if recording_duration > 0:
        fps = frame_count / recording_duration
    else:
        fps = 0
    print(f"录制时长: {recording_duration:.2f}秒，帧率: {fps:.2f} FPS")
    # 获取第一张图像以获取宽高
    first_image_path = os.path.join(save_photo_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编写器
    save_video_dir = os.path.join(os.getcwd(), "AIdjango", "dist", "livedisplay_record2video")
    os.makedirs(save_video_dir, exist_ok=True)  # 确保输出目录存在
    video_name = os.path.join(save_video_dir, latest_folder + '.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码方式
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 编码
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
    # fourcc = cv2.VideoWriter_fourcc(*'hevc')

    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))  

    # 读取并写入图像到视频
    for image_file in image_files:
        image_path = os.path.join(save_photo_dir, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频编写器
    video_writer.release()
    print(f"视频已成功保存为: {video_name}")

def stream_record_download(request):
        # print(123123)
        # print(request.body)
        # data = json.loads(request.body)
        # video_name = data.get('name')
        video_name = request.GET.get('name') 
        file_path =  f'AIdjango/dist/livedisplay_record2video/{video_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response

def stream_video_download(request):
        # data = json.loads(request.body)
        # video_name = data.get('name')
        # video_name = urllib.parse.quote(video_name)
        video_name = request.GET.get('name')
        # video_name = "2024-09-29-21-36-45.avi"
        file_path =  f'AIdjango/dist/UploadvideoProcess/{video_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response

def stream_photo_download(request):
        data = json.loads(request.body)
        photo_name = data.get('name')
        photo_name = urllib.parse.quote(photo_name)
        # photo_name = "6bd979a269cb070014f1a1a71e90e364.png"
        file_path =  f'AIdjango/dist/UploadphotoProcess/{photo_name}'
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response


    
def video(request):

    global camera
    if camera is not None:
        return StreamingHttpResponse(gen_display(camera), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return JsonResponse({'status': 'Camera open,please open'})





def getAllRecordFile(request):
    base_dir = 'AIdjango/dist/livedisplay_record2video'
    files = [urllib.parse.unquote(f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})

def getAllVideoFile(request):
    base_dir = 'AIdjango/dist/UploadvideoProcess'
    files = [urllib.parse.unquote(f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})

def getAllPhotoFile(request):
    base_dir = 'AIdjango/dist/UploadphotoProcess'
    files = [urllib.parse.unquote(f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return JsonResponse({'files': files})

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

def upload_video(request):
    video_url = None  # 初始化为 None，防止首次加载时报错
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            Vname = video.name
            # Vname = urllib.parse.quote(Vname)
            name, ext = os.path.splitext(Vname)
            file_path = f'AIdjango/dist/UploadvideoSave/{Vname}'
            counter = 1
            while os.path.exists(file_path):
                file_path = f'AIdjango/dist/UploadvideoSave/{name}_{counter}{ext}'
                counter += 1
            videoname = os.path.basename(file_path)
            print(videoname)
            print(file_path)
            try:
                with open(file_path, 'wb') as f:
                    for chunk in video.chunks():
                        f.write(chunk)
            
                # threading.Thread(target=video_detection, args=(videoname,)).start()
            except Exception as e:
                return JsonResponse({'message': "Failed to process video", 'error': str(e)}, status=500)
            return JsonResponse({'message': "upload finish",  "videoname":urllib.parse.unquote(videoname),'success': 1}, status=200)

        else:
            return JsonResponse({'message': "No video file uploaded.", 'success': 0}, status=400)

    return JsonResponse({'message': "please use post",  'success': 0}, status=200)

def start_process_video(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        video_name = data.get("name")
        print(video_name)
        threading.Thread(target=video_detection, args=(video_name,)).start()
        return JsonResponse({'message': "processing",  'success': 0}, status=200)
    return JsonResponse({'message': "please use post",  'success': 0}, status=200)

def video_detection(video_name):
    global queueISdeal

    background_thread = threading.Thread(target=background_processing, daemon=True)
    background_thread.start()
    current_dir = os.getcwd()
    video_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadvideoSave', video_name)
    print(video_path)
    video_name = urllib.parse.quote(video_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    cache.set(video_name, 0)
    cache.set(video_name+"total", 0)
    cache.set(video_name+"current", 0)
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 创建保存处理后的视频的文件名
    
    processed_video_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadvideoProcess',video_name )

    # 创建 VideoWriter 对象
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 编码
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
    max_num_defects = 0
    all_detected_labels = set()
    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        if not ret:
            break
        img = frame
        defect_info={}
        if params["dark"]:
            print(2)
        if params["CLAHE"]:
            img = CLAHE(img)
        if params["detail"]:
            img = detail(img)
        if params["Sharpening"]:
            img = Sharpening(img)
        if params["size"]:
            result= BigModel.predict(img)
        else:
            result = SmallModel.predict(img)
        label_map = result['label_map']
        unique_labels = np.unique(label_map)  # 获取所有唯一标签值
        num_defects = len(unique_labels) - 1  # 减去背景类别
        
        max_num_defects = max(max_num_defects, num_defects)
        all_detected_labels.update(unique_labels.tolist())
        defect_info["num_defects"] = max_num_defects
        defect_info["detected_labels"] = sorted(all_detected_labels)  # 将标签转换为列表以便保存
        vis_result = pdx.seg.my_visualize(frame, result, weight=0.4,color=[0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255])

        # 处理图像并获取结果
        # if params["haze_enabled"]:
        #     frame = haze_net.haze_frame(frame)

        # if params["dark_enabled"]:
        #     frame = dark_net.process_frame(frame)
        # if params["seg_enable"]:
        #     frame = seg_net.process_frame(frame)#传入RGB，传出RGB
        # frame = paddledetection_net.predit(frame)
        # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame= cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
        # 将处理后的帧写入新的视频文件
        out.write(vis_result)
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cache.set(video_name+"total",  total_frames)  
        cache.set(video_name+"current",  current_frame)  
        cache.set(video_name, current_frame / total_frames * 100)  
    photo_instance = Photo(
                photo_name=video_name,  # 照片名称
                photo_path=video_path,
                source = "video",
                result_path=processed_video_path,  # 处理后的照片路径
                process_type=determine_process_type(params),  # 处理类型（你可以在params中加入多个处理类型）
                upload_time=timezone.now(),  # 上传时间（假设这里使用当前时间）
                process_time=timezone.now(),  # 处理时间（假设这里使用当前时间）
                defect_info=defect_info  # 缺陷信息（包括快速剔除和其他处理类型）
                        )

    photo_instance.save()
    # 释放资源
    cap.release()
    out.release()
    queueISdeal = False
    


def get_progress(request):
    print(request)
    data = json.loads(request.body)
    video_name = data.get("video_name")
    video_name = urllib.parse.quote(video_name)
    if(video_name):
        progress = cache.get(video_name, 0)
        current = cache.get(video_name+"current", 0)
        total = cache.get(video_name+"total", 0)
        return JsonResponse({'progress': progress,"current":current,"total":total})
    return JsonResponse({'progress': "not process"})

def upload_photo(request):
    photo_url = None  # 初始化为 None，防止首次加载时报错
    print(request.method)
    if request.method == 'POST':
        if 'photo' in request.FILES:
            photo = request.FILES['photo']
            Pname = photo.name
            # Pname = urllib.parse.quote(Pname)
            file_path = f'AIdjango/dist/UploadphotoSave/{Pname}'
            name, ext = os.path.splitext(Pname)

            counter = 1
            while os.path.exists(file_path):
                file_path = f'AIdjango/dist/UploadphotoSave/{name}_{counter}{ext}'
                counter += 1
            photoname = os.path.basename(file_path)
            try:
                # 将上传的照片保存到指定路径
                with open(file_path, 'wb') as f:
                    for chunk in photo.chunks():
                        f.write(chunk)
            except Exception as e:
                print(f"Failed to upload photo: {e}")
            return JsonResponse({'message': "sucess post",  "photoname":urllib.parse.unquote(photoname),'success': 1}, status=200)
        else:
            print("No photo file uploaded.")

    return JsonResponse({'message': "please use post",  "photoname":'','success': 0}, status=200)

def start_process_photo(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        photo_name = data.get("name")
        print(photo_name)
        threading.Thread(target=photo_processing, args=(photo_name,)).start()
        return JsonResponse({'message': "processing",  'success': 0}, status=200)
    return JsonResponse({'message': "please use post",  'success': 0}, status=200)

import cv2
import numpy as np
from .models import Photo
from django.utils import timezone



def photo_processing(photo_name):
    # print(photo_name)
    # 00d7ae946.jpg
    global queueISdeal
    global queueISdeal
    background_thread = threading.Thread(target=background_processing, daemon=True)
    background_thread.start()
    current_dir = os.getcwd()
    photo_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoSave', photo_name)
    photo_path_Process = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoProcess')
    photo_path_Process_path = os.path.join(current_dir, 'AIdjango', 'dist', 'UploadphotoProcess', urllib.parse.quote(photo_name))
    
    # 检查文件是否存在
    if not os.path.exists(photo_path):
        print(f"Error: File does not exist at {photo_path}")
        return
    pil_img = Image.open(photo_path)
    img = np.array(pil_img)  # 转换为NumPy数组
    has_defect = True
    defect_info = {}
    if params["fastElimination"]:
        has_defect, diff_map = fastElimination(img)
    if has_defect:
        img = img[:, :, ::-1]  # 将RGB转换为BGR格式
        if params["dark"]:
            # img = deblurring(img)
            print(1)
        if params["CLAHE"]:
            img = CLAHE(img)
        if params["detail"]:
            img = detail(img)
        if params["Sharpening"]:
            img = Sharpening(img)
        print(img.shape)
        if params["size"]:
            result = BigModel.predict(img)
        else:
            result = SmallModel.predict(img)
        label_map = result['label_map']
        unique_labels = np.unique(label_map)  # 获取所有唯一标签值
        num_defects = len(unique_labels) - 1  # 减去背景类别
        defect_info["num_defects"] = num_defects
        defect_info["detected_labels"] = unique_labels.tolist()  # 将标签转换为列表以便保存

        visualize_path = os.path.join(photo_path_Process, f"{photo_name}")
        pdx.seg.my_visualize(photo_path, result, weight=0.4, save_dir=visualize_path,color=[0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255])
    else:
        # img.save(photo_path_Process_path)  # 保存图像
        processed_img = Image.fromarray(img)  # 将 NumPy 数组转回 PIL 图像
        processed_img.save(photo_path_Process_path)  # 保存图像到指定路径
    photo_instance = Photo(
        photo_name=photo_name,  # 照片名称
        photo_path=photo_path,  # 原始照片路径
        source = "photo",
        result_path=photo_path_Process_path,  # 处理后的照片路径
        process_type=determine_process_type(params),  # 处理类型（你可以在params中加入多个处理类型）
        upload_time=timezone.now(),  # 上传时间（假设这里使用当前时间）
        process_time=timezone.now(),  # 处理时间（假设这里使用当前时间）
        defect_info=defect_info  # 缺陷信息（包括快速剔除和其他处理类型）
    )

    # 保存到数据库
    photo_instance.save()
    # 调用模型进行预测
    # result = BigModel.predict(img)
    # label_map = result['label_map']  # 分割后的标签图
    # score_map = result['score_map']  # 分割的得分图
    # 保存label_map和score_map
    # label_map_path = os.path.join(photo_path_Process, f"label_map_{photo_name}")
    # score_map_path = os.path.join(photo_path_Process, f"score_map_{photo_name}")
    
    # label_map_img = Image.fromarray((label_map*255).astype(np.uint8))
    # label_map_img.save(label_map_path)

    # score_map_img = Image.fromarray((score_map ).astype(np.uint8))  # 将得分图归一化到0-255
    # print(score_map.shape)
    # score_map_img.save(score_map_path)

    # 可视化分割结果并保存


    conversion_thread = threading.Thread(target=convert_image_format, args=(photo_name, "AIdjango/dist/UploadphotoProcess"))
    conversion_thread.start()  # 启动线程
    queueISdeal =  False

# 类别	名称	BGR颜色	说明
# 0	背景	[0, 0, 0]	黑色
# 1	缺陷类型1	[0, 0, 255]	红色
# 2	缺陷类型2	[0, 255, 0]	绿色
# 3	缺陷类型3	[255, 0, 0]	蓝色
# 4	缺陷类型4	[0, 255, 255]	黄色（青+红）
def convert_image_format(photo_name, directory):
    # 对照片名称进行URL编码
    encoded_name = urllib.parse.quote(photo_name)
    
    # 获取文件的完整路径
    file_path = os.path.join(directory, encoded_name)
    name = os.path.splitext(photo_name)[0]  # 获取文件名（不包含扩展名）
    
    # 尝试查找 .jpg 或 .png 文件
    jpg_file_path = os.path.join(directory, f"{name}.jpg")
    png_file_path = os.path.join(directory, f"{name}.png")
    print(jpg_file_path,png_file_path)
    # 检查文件是否存在
    while True:
        if os.path.exists(jpg_file_path):
            # 如果找到 JPG 文件，则将其转换为 PNG 格式
            try:
                with Image.open(jpg_file_path) as img:
                    new_png_file_path = os.path.join(directory, f"{name}.png")
                    img.save(new_png_file_path, 'PNG')
                    print(f"Converted {jpg_file_path} to {new_png_file_path}")
                    break
            except Exception as e:
                print(f"Error converting {jpg_file_path} to PNG: {e}")
                break
        
        elif os.path.exists(png_file_path):
            # 如果找到 PNG 文件，则将其转换为 JPG 格式
            try:
                with Image.open(png_file_path) as img:
                    new_jpg_file_path = os.path.join(directory, f"{name}.jpg")
                    img.convert("RGB").save(new_jpg_file_path, 'JPEG')
                    print(f"Converted {png_file_path} to {new_jpg_file_path}")
                    break
            except Exception as e:
                print(f"Error converting {png_file_path} to JPG: {e}")
                break
                    
        
        else:
            print(f"No .jpg or .png file found with the name {name} in the directory.")

        






def video_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        video_name = data.get("video_name")
        
    else:
        video_name = None
    
    video_url = f'/media/{video_name}'
    # return render(request, 'showvideo.html', {'video_url': video_url})
    return JsonResponse({'video_url': video_url,"success":1})

def photo_view(request):
    # 视频文件的 URL
    if request.body:
        data = json.loads(request.body)
        photo_name = data.get("photo_name")
    else:
        photo_name = None
    photo_url = f'/media/{photo_name}'
    # return render(request, 'showphoto.html', {'photo_url': photo_url})
    # delete_path = os.path.join(settings.MEDIA_ROOT,photo_name)
    # os.remove(delete_path)
    return JsonResponse({'photo_url': photo_url,"success":1})

# def stream_video(request):
#     video_path = "AIdjango/dist/UploadvideoSave/show.mp4"  # 替换为视频文件的实际路径
#     wrapper = FileWrapper(open(video_path, 'rb'))
#     response = HttpResponse(wrapper, content_type='video/mp4')
#     response['Content-Length'] = os.path.getsize(video_path)
#     return response

def stream_video(request):
    name = "qa.mp4"
    style = request.GET.get("style")
    
    if style is not None:
        style = int(style)  # 将 style 转换为整数
    else:
        return HttpResponse(status=400)  # 错误请求

    if style == 1:
        video_path = "AIdjango/dist/livedisplay_record2video/" + name
    elif style == 2:
        name = urllib.parse.quote(name)
        video_path = "AIdjango/dist/UploadvideoProcess/" + name
    elif style == 3:
        video_path = "AIdjango/dist/UploadvideoSave/" + name
    else:
        return HttpResponse(status=400)  # 不支持的样式

    print(video_path)
    range_header = request.META.get('HTTP_RANGE', '').strip()
    if os.path.exists(video_path):
        if range_header:
            size = os.path.getsize(video_path)
            start, end = parse_range_header(range_header, size)

            if start is None or end is None:
                return HttpResponse(status=416)

            if start >= size or end >= size:
                return HttpResponse(status=416)

            length = end - start + 1
            file = open(video_path, 'rb')
            file.seek(start)
            wrapper = FileWrapper(file)
            response = HttpResponse(wrapper, content_type='video/mp4', status=206)
            response['Content-Disposition'] = 'inline'
            response['Content-Length'] = str(length)
            response['Content-Range'] = f'bytes {start}-{end}/{size}'
            return response

        wrapper = FileWrapper(open(video_path, 'rb'))
        response = HttpResponse(wrapper, content_type='video/mp4')
        response['Content-Length'] = os.path.getsize(video_path)
        return response
    else:
        return JsonResponse({'message': "filepath is not exist",  'success': 0}, status=200)

def parse_range_header(range_header, size):
    if range_header:
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start) if start else 0
        end = int(end) if end else size - 1
        return start, end
    return None, None



def stream_photo(request):
    name = request.GET.get("name")
    style = request.GET.get("style")
    print(name)
    
    if style is not None:
        style = int(style)  # 将 style 转换为整数
    else:
        return HttpResponse(status=400)  # 错误请求
    
    if style == 1:
        image_path = "AIdjango/dist/UploadphotoSave/" + name
    elif style == 2:
        name = urllib.parse.quote(name)
        image_path = "AIdjango/dist/UploadphotoProcess/" + name

    else:
        return HttpResponse(status=400)  # 不支持的样式

    print(image_path)

    if os.path.exists(image_path):
        try:
            # 使用 Pillow 打开图像
            with Image.open(image_path) as pil_img:
                # 创建一个 BytesIO 对象来保存图像
                img_byte_array = io.BytesIO()
                pil_img.save(img_byte_array, format='JPEG')  # 将图像保存为 JPEG 格式
                img_byte_array.seek(0)  # 移动到 BytesIO 的开始位置
                return HttpResponse(img_byte_array.getvalue(), content_type='image/jpeg')
        except Exception as e:
            print(f"Failed to load image with Pillow: {e}")
            return HttpResponse(status=500)  # 服务器错误
    else:
        return JsonResponse({'message': "filepath is not exist",  'success': 0}, status=200)
    

def clear_directory(directory):
    try:
        # 列出目录下的所有文件
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # 检查是否为文件，并删除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        print("Directory cleared successfully.")
    except Exception as e:
        print(f"Error clearing directory: {e}")
import numpy as np
from django.http import JsonResponse

def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return [convert_to_serializable(value) for value in data.values()]  # 只保留值，形成数组
    elif isinstance(data, np.float32):
        return float(data)  # 转换 float32 为 float
    return data

def log(request):
    if request.method == 'POST':
        response_data = []
        try:
            # 获取前端传来的 JSON 数据
            data = json.loads(request.body)
            print(0000000000000000000000)
            source = data.get('source')

            # 根据 source 字段过滤符合条件的照片记录
            if source == 'photo':
                photo_instances = Photo.objects.filter(source='photo')
            else:
                return JsonResponse({'error': 'Invalid source value'}, status=400)

            # 如果没有找到符合条件的记录
            if not photo_instances:
                return JsonResponse({'error': 'No photos found with the specified source'}, status=404)

            # 返回符合条件的照片信息
            
            for photo_instance in photo_instances:
                photo_data = {
                    'photo_name': photo_instance.photo_name,
                    'photo_path': photo_instance.photo_path,
                    'result_path': photo_instance.result_path,
                    'process_type': photo_instance.process_type,
                    'upload_time': photo_instance.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'process_time': photo_instance.process_time.strftime('%Y-%m-%d %H:%M:%S') if photo_instance.process_time else None,
                    'defect_info': photo_instance.defect_info,  # 缺陷信息
                }
                response_data.append(photo_data)

            # 返回符合条件的照片数据
            print(122131231213)
            return JsonResponse({'photos': response_data})

        except json.JSONDecodeError:
            # return JsonResponse({'error': 'Invalid JSON format'}, status=400)
            return JsonResponse({'photos': response_data})

    else:
        
        return JsonResponse({'error': 'Invalid request method'}, status=400)
from django.http import JsonResponse
from .models import Photo
from django.db.models import Q
import json
from datetime import datetime

def query(request):
    if request.method == 'POST':
        try:
            # 获取前端传来的 JSON 数据
            data = json.loads(request.body)
            print(data)
            # 提取查询条件
            source = data.get('source', None)
            photo_name = data.get('photo_name', None)
            start_time = data.get('start_time', None)
            end_time = data.get('end_time', None)
            num_defects = data.get('num_defects', None)
            detected_labels = data.get('detected_labels', None)

            # 初始化查询条件
            query_conditions = Q()

            # 如果提供了 source 参数，进行筛选
            if source:
                query_conditions &= Q(source=source)

            # 如果提供了照片名称，进行筛选
            if photo_name:
                query_conditions &= Q(photo_name__icontains=photo_name)

            # 如果提供了时间段，进行筛选
            if start_time:
                try:
                    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                    query_conditions &= Q(upload_time__gte=start_time)
                except ValueError:
                    return JsonResponse({'error': 'Invalid start_time format, expected "YYYY-MM-DD HH:MM:SS"'}, status=400)

            if end_time:
                try:
                    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                    query_conditions &= Q(upload_time__lte=end_time)
                except ValueError:
                    return JsonResponse({'error': 'Invalid end_time format, expected "YYYY-MM-DD HH:MM:SS"'}, status=400)

            # 如果提供了缺陷信息（num_defects），进行筛选
            if num_defects:
                try:
                    num_defects = int(num_defects)
                    query_conditions &= Q(defect_info__num_defects=num_defects)
                except ValueError:
                    return JsonResponse({'error': 'Invalid num_defects, it must be an integer'}, status=400)

            # 如果提供了缺陷标签（detected_labels），进行筛选
            # if detected_labels:
            #         try:
            #             # 将 detected_labels 字符串转换为列表
            #             detected_labels = list(map(int, detected_labels.split(',')))
            #             print(detected_labels)  # 输出 [0, 3]

            #             # 初始化查询条件，逐个检查 `detected_labels` 中的每个标签
            #             for label in detected_labels:
            #                 # 确保每个标签都在 defect_info__detected_labels 中
            #                 query_conditions &= Q(defect_info__detected_labels__exact=label)

            #         except ValueError:
            #             return JsonResponse({'error': 'Invalid detected_labels format, it must be a comma-separated list of integers'}, status=400)


            # 根据查询条件过滤照片记录
            photo_instances = Photo.objects.filter(query_conditions)

            # 如果没有找到符合条件的记录
            if not photo_instances:
                return JsonResponse({'photos': []})

            # 返回符合条件的照片信息
            response_data = []
            for photo_instance in photo_instances:
                photo_data = {
                    'photo_name': photo_instance.photo_name,
                    'photo_path': photo_instance.photo_path,
                    'result_path': photo_instance.result_path,
                    'process_type': photo_instance.process_type,
                    'upload_time': photo_instance.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'process_time': photo_instance.process_time.strftime('%Y-%m-%d %H:%M:%S') if photo_instance.process_time else None,
                    'defect_info': photo_instance.defect_info,  # 缺陷信息
                }
                response_data.append(photo_data)

            # 返回符合条件的照片数据
            return JsonResponse({'photos': response_data})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Invalid request method, only POST allowed'}, status=400)


# {
# "photo_name": "0aa7955fd.jpg",
#   "source": "photo",
#   "start_time": "2025-04-08 17:08:00",
# "num_defects": 1,
#   "end_time": "2025-04-08 17:08:09"
# }


import os
from django.http import JsonResponse

def list_photos(request):
    style = request.GET.get("style")
    
    if style is None:
        return JsonResponse({'message': "Missing style", 'success': 0}, status=400)

    try:
        style = int(style)
    except ValueError:
        return JsonResponse({'message': "Invalid style", 'success': 0}, status=400)

    # 选择文件夹路径
    if style == 1:
        folder_path = "AIdjango/dist/UploadphotoSave"
    elif style == 2:
        folder_path = "AIdjango/dist/UploadphotoProcess"
    else:
        return JsonResponse({'message': "Unsupported style", 'success': 0}, status=400)

    if not os.path.exists(folder_path):
        return JsonResponse({'message': "Folder not found", 'success': 0}, status=404)

    # 只返回图片文件名
    photo_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    return JsonResponse({
        'success': 1,
        'photos': photo_list
    })

import zipfile
import io
import os
from django.http import HttpResponse

def download_all_photos(request):
    style = request.GET.get("style")
    try:
        style = int(style)
    except (TypeError, ValueError):
        return HttpResponse(status=400)

    if style == 1:
        folder_path = "AIdjango/dist/livedisplay_recordphoto"
    elif style == 2:
        folder_path = "AIdjango/dist/UploadphotoProcess"
    else:
        return HttpResponse(status=400)

    if not os.path.exists(folder_path):
        return HttpResponse("Folder not found", status=404)

    # 将所有图片打包成 zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, filename)
                zip_file.write(file_path, arcname=filename)

    zip_buffer.seek(0)
    return HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
def download_all_video(request):
    style = request.GET.get("style")
    try:
        style = int(style)
    except (TypeError, ValueError):
        return HttpResponse(status=400)

    if style == 1:
        folder_path = "AIdjango/dist/livedisplay_record2video"
    elif style == 2:
        folder_path = "AIdjango/dist/UploadvideoProcess"
    else:
        return HttpResponse(status=400)

    if not os.path.exists(folder_path):
        return HttpResponse("Folder not found", status=404)

    # 打包所有视频文件
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(folder_path, filename)
                zip_file.write(file_path, arcname=filename)

    zip_buffer.seek(0)
    return HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
