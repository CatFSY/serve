from django.db import models
from django.utils import timezone
import json

class Photo(models.Model):
    # 照片唯一标识符
    photo_name = models.CharField(max_length=255)  # 照片名称（包括扩展名）
    
    # 照片路径
    photo_path = models.CharField(max_length=1024)  # 照片未处理前的路径（原始照片）
    result_path = models.CharField(max_length=1024)  # 照片处理后的路径
    source = models.CharField(max_length=10)  # 照片处理后的路径
    
    # 照片处理类型
    PROCESS_TYPE_CHOICES = [
        ('deblurring', 'deblurring'),
        ('detail', 'detail'),
        ('Sharpening', 'Sharpening'),
        ('fastElimination', 'fastElimination'),
        ('CLAHE', 'CLAHE'),
        ('big','big'),
        ('small','small'),
        
    ]

    process_type = models.CharField(
        max_length=100,  # 扩大长度，以适应多个处理类型
        choices=PROCESS_TYPE_CHOICES,
        blank=True,  # 允许为空，避免未选时出现问题
    )

    
    # 上传时间和处理时间
    upload_time = models.DateTimeField(default=timezone.now)  # 上传时间
    process_time = models.DateTimeField(null=True, blank=True)  # 处理时间

    # 缺陷信息，存储为 JSON 格式
    defect_info = models.JSONField(null=True, blank=True)  # 缺陷信息
    
    def __str__(self):
        return f"Photo {self.photo_name} (ID: {self.id})"
    
    class Meta:
        verbose_name = 'Photo'
        verbose_name_plural = 'Photos'
        ordering = ['-upload_time']  # 按照上传时间降序排列


