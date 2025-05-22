from django.contrib import admin
from django.urls import path
from user.gen_display import video,upload_video,upload_photo,ConfirmParams, close_camera, open_camera,getAllRecordFile,get_progress
from user.gen_display import initialize,getAllPhotoFile,getAllVideoFile,getAllCam, Camchoice,video_record_on,video_record_off,stream_record_download,query,list_photos
from user.gen_display import stream_video_download,stream_photo_download,stream_video,stream_photo,start_process_video,start_process_photo,log,get_sizes,delete,index,download_all_photos,download_all_video
from django.conf import settings
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path("",index) ,        #初始化深度学习模型
    path("ConfirmParams",ConfirmParams),
    path("getAllCam",getAllCam),#得到所有的摄像头设备
    path("Camchoice",Camchoice),#给出使用的摄像头
    path('opencam', open_camera),     #打开摄像头
    path('closecam',  close_camera),     #关闭摄像头
    path('livedisplay', video),     #实时演示功能
    path('video_record_on', video_record_on),     #开启录制
    path('video_record_off',  video_record_off),    #关闭录制
    path("getAllRecordFile",getAllRecordFile),#得到所有的录制文件
    path('uploadVideo', upload_video),#上传视频
    path('uploadPhoto', upload_photo),#上传照片
    path("get_progress",get_progress),#得到视频处理的进度条
    path("stream_record_download",stream_record_download),#下载录制的视频。
    path("stream_photo_download",stream_photo_download),#下载录制的视频。
    path("stream_video_download",stream_video_download),#下载录制的视频。
    path('upload_photo', upload_photo, name='upload_photo'),#上传照片
    path("stream_video",stream_video),
    path("stream_record",stream_record_download),
    path("stream_photo",stream_photo),
    path("start_process_video",start_process_video),
    path("start_process_photo",start_process_photo),
    path("log",log),
    path("get_sizes",get_sizes),
    path("delete",delete),
    path("getAllPhotoFile",getAllPhotoFile),#得到所有的照片文件
    path("getAllVideoFile",getAllVideoFile),#得到所有的视频文件
    path("api/ConfirmParams", ConfirmParams),
    path("api/getAllCam", getAllCam),  # 得到所有的摄像头设备
    path("api/Camchoice", Camchoice),  # 给出使用的摄像头
    path('api/opencam', open_camera),  # 打开摄像头
    path('api/closecam', close_camera),  # 关闭摄像头
    path('api/livedisplay', video),  # 实时演示功能
    path('api/video_record_on', video_record_on),  # 开启录制
    path('api/video_record_off', video_record_off),  # 关闭录制
    path("api/getAllRecordFile", getAllRecordFile),  # 得到所有的录制文件
    path('api/uploadVideo', upload_video),  # 上传视频
    path('api/uploadPhoto', upload_photo),  # 上传照片
    path("api/get_progress", get_progress),  # 得到视频处理的进度条
    path("api/stream_record_download", stream_record_download),  # 下载录制的视频
    path("api/stream_photo_download", stream_photo_download),  # 下载录制的视频
    path("api/stream_video_download", stream_video_download),  # 下载录制的视频
    path('api/upload_photo', upload_photo, name='upload_photo'),  # 上传照片
    path("api/stream_video", stream_video),
    path("api/stream_record", stream_record_download),
    path("api/stream_photo", stream_photo),
    path("api/start_process_video", start_process_video),
    path("api/start_process_photo", start_process_photo),
    path("api/log", log),
    path("api/get_sizes", get_sizes),
    path("api/delete", delete),

    path("api/getAllPhotoFile", getAllPhotoFile),  # 得到所有的照片文件
    path("api/getAllVideoFile", getAllVideoFile),  # 得到所有的视频文件
    path("api/query", query),  # 得到所有的视频文件
    path("query", query),  # 得到所有的视频文件
    path("list_photos",list_photos),
    path("download_all_photos",download_all_photos),
    path("api/download_all_video",download_all_video),
    path("api/download_all_photos",download_all_photos),
    path("download_all_video",download_all_video)


]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
