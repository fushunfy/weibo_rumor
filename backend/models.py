# from django.db import models
#
#
# # Create your models here.
#
# class UserModel(models.Model):
#     username = models.CharField(max_length=20, unique=True)  # unique代表用户名唯一
#     password = models.CharField(max_length=20)
#     email = models.EmailField(max_length=50)
#
#     def __str__(self):
#         return self.username
#
#
# class FileModel(models.Model):
#     uploadFile = models.FileField(upload_to='fileDir')
