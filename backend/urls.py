from django.urls import path
from . import views
urlpatterns = [
    path('api/', views.login),
    path('api/register', views.register),
    path('api/home/uploadFile', views.uploadFile),
    path('api/home/runPredictFile', views.runPredictFile)
]