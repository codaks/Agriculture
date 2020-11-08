from . import views
from django.urls import include,path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns



urlpatterns = [
    path("",views.returnRedirect, name = "returnRedirect"),
    path("home", views.index, name = "home"),
    path("live",views.live, name = "live"),
    path("tomato-area", views.tomato, name = "tomato-area"),
]
