import json
from rest_framework import serializers

from msp.models import Scenario, ResultScenario, Document


class DocumentPathSerializer(serializers.StringRelatedField):
    """From Document Model keep only value field from __str__ method"""
    def to_internal_value(self, value):
        return value


class DocumentSerializer(serializers.ModelSerializer):
    """Model Serializer for Document Model"""
    class Meta:
        model = Document
        fields = ('id', 'filename', 'file')
        read_only = ('id',)


class ScenarioSerializer(serializers.ModelSerializer):
    """Model Serializer for Scenario Model"""

    # Document Model are converted into their string related filed
    dataset = DocumentPathSerializer()
    pipeline = DocumentPathSerializer()

    # Transform string field is parsed into JSON object with get_transforms method
    transforms = serializers.SerializerMethodField()

    class Meta:
        model = Scenario
        fields = '__all__'
        read_only = ('id',)

    def get_transforms(self, obj):
        if obj.transforms:
            return json.loads(obj.transforms)
        else:
            return None


class ResultScenarioSerializer(serializers.ModelSerializer):
    """Model Serializer for Result Scenario Model"""
    class Meta:
        model = ResultScenario
        fields = '__all__'
        read_only = ('id',)

