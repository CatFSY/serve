import random
import os
import datetime, time
from pathlib import Path
from rest_framework.response import Response
from django.conf import settings

def img_proccess_save(image, file):
    # 防重名
    name = Path(image.name)
    img_pure_name = name.stem + '_' + str(int(time.time())) # name.stem：提取无后缀的文件名
    img_extend_name = name.suffix # 提取后缀名
    img_name = img_pure_name + img_extend_name # 新的文件名

    # 创建存储路径
    img_dir = os.path.join(settings.MEDIA_ROOT, file) # 想要保存的文件夹 
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    img_file_year = os.path.join(img_dir, datetime.datetime.now().strftime("%Y")) # 按年保存的文件夹
    if not os.path.exists(img_file_year):
        os.mkdir(img_file_year)
    img_file_month = os.path.join(img_file_year, datetime.datetime.now().strftime("%m")) # 按月保存的文件夹
    if not os.path.exists(img_file_month):
        os.mkdir(img_file_month)

    # 存储图片
    destination = open(os.path.join(img_file_month, img_name), 'wb+')
    for chunk in image.chunks(): # 对图片切片
        destination.write(chunk) # 把切片写入
    destination.close()

    # 传回给后端ImageField要存储的图片路径
    backend_relative_path = file + '/' + datetime.datetime.now().strftime("%Y") + '/' + datetime.datetime.now().strftime("%m") + '/' + img_name 

    return img_name, backend_relative_path