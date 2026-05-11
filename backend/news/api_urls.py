from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.predict_news, name="predict"),
    path("history/", views.prediction_history, name="history"),
    path("search/", views.search_news_api, name="search"),
    path("auth/register/", views.register_user, name="register"),
    path("auth/login/", views.login_user, name="login"),
    path("auth/logout/", views.logout_user, name="logout"),
    path("auth/me/", views.auth_me, name="auth_me"),
]
