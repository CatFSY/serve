# Generated by Django 4.2.20 on 2025-04-08 16:50

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Photo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo_name', models.CharField(max_length=255)),
                ('photo_path', models.CharField(max_length=1024)),
                ('result_path', models.CharField(max_length=1024)),
                ('source', models.CharField(max_length=10)),
                ('process_type', models.CharField(blank=True, choices=[('deblurring', 'deblurring'), ('detail', 'detail'), ('Sharpening', 'Sharpening'), ('fastElimination', 'fastElimination'), ('CLAHE', 'CLAHE'), ('big', 'big'), ('small', 'small')], max_length=100)),
                ('upload_time', models.DateTimeField(default=django.utils.timezone.now)),
                ('process_time', models.DateTimeField(blank=True, null=True)),
                ('defect_info', models.JSONField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Photo',
                'verbose_name_plural': 'Photos',
                'ordering': ['-upload_time'],
            },
        ),
    ]
