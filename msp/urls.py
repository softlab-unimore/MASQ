from django.urls import path

from . import views

app_name = 'msp'

urlpatterns = [
    path('', views.index, name='index'),
    path('restart/', views.restart, name='restart'),
]
