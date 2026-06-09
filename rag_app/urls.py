from django.urls import path
from rag_app import views

urlpatterns = [
    path('',                          views.index,          name='index'),
    path('upload/',                   views.upload,         name='upload'),
    path('chat/',                     views.chat,           name='chat'),
    path('docs/status/',              views.docs_status_all,name='docs_status_all'),
    path('docs/<uuid:doc_id>/status/',views.doc_status,     name='doc_status'),
    path('docs/<uuid:doc_id>/index/', views.doc_index,      name='doc_index'),
    path('docs/<uuid:doc_id>/delete/',views.doc_delete,     name='doc_delete'),
]
