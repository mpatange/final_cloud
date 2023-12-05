from django.urls import path
from . import views
from .views import home,predict,random

urlpatterns = [
    path('',views.home,name='home'),
    path('predict/',views.predict,name='predict'),
    path('random/',views.random,name='random'),
]