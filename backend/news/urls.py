from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('results/', views.results_page, name='results'),
    path('about/', views.about_page, name='about'),
    path('auth/', views.auth_page, name='auth'),
    path('upload/', views.upload_page, name='upload'),
]