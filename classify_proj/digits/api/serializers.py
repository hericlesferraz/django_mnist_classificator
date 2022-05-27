from rest_framework import serializers
from ..models import Digit
from django.core.files.base import ContentFile
import base64
import uuid

class Base64ImageField(serializers.ImageField):

    def to_internal_value(self, data):
        _format, str_img = data.split(';base64')
        decoded_file = base64.b64decode(str_img)
        fname = f"str(uuid.uuid64())[:10].png"
        data = ContentFile(decoded_file, name=fname)

        return super().to_internal_value(data)

class DigitSerializer(serializers.ModelSerializer):
    image = Base64ImageField()
    class Meta:
        model = Digit
        fields = ('id', 'image', 'result')