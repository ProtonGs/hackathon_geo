import uuid
from django.db import models


class Document(models.Model):
    STATUS = [
        ('pending',  'Ожидает'),
        ('indexing', 'Индексируется'),
        ('indexed',  'Готов'),
        ('error',    'Ошибка'),
    ]

    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name         = models.CharField(max_length=500)
    file         = models.FileField(upload_to='pdfs/')
    status       = models.CharField(max_length=20, choices=STATUS, default='pending')
    progress     = models.TextField(blank=True, default='')
    error_msg    = models.TextField(blank=True, default='')
    text_chunks  = models.IntegerField(default=0)
    images_count = models.IntegerField(default=0)
    total_pages  = models.IntegerField(default=0)
    uploaded_at  = models.DateTimeField(auto_now_add=True)
    indexed_at   = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.name

    def to_dict(self):
        return {
            'id':           str(self.id),
            'name':         self.name,
            'status':       self.status,
            'progress':     self.progress,
            'error_msg':    self.error_msg,
            'text_chunks':  self.text_chunks,
            'images_count': self.images_count,
            'total_pages':  self.total_pages,
            'indexed_at':   self.indexed_at.isoformat() if self.indexed_at else None,
        }
