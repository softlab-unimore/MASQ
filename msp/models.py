import pandas as pd
from django.db import models

from dbconnection.connector import get_tables, get_table


class Scenario(models.Model):
    is_db = models.BooleanField()
    # dataset = models.FileField(blank=True, upload_to='dataset/')
    dataset = models.ForeignKey('Document', on_delete=models.SET_NULL, null=True, blank=True, related_name='dataset')
    db_url = models.CharField(max_length=256, blank=True, null=True)
    table = models.CharField(max_length=256, blank=True, null=True)

    # tables = models.CharField(max_length=256, blank=True)
    # columns = models.CharField(max_length=256, )

    mode = models.CharField(max_length=256, choices=[
        ('train', 'Train'),
        ('test', 'Test'),
    ])

    labels_type = models.CharField(max_length=256, blank=True, null=True, choices=[
        ('column', 'Column'),
        ('table', 'Table'),
        ('file', 'File'),
    ])
    labels = models.CharField(max_length=256, blank=True, null=True)
    # labelsFile = models.FileField(blank=True, upload_to='labels/')
    # labelsColumn = models.CharField(max_length=256, blank=True)
    # labelsTable = models.CharField(max_length=256, blank=True)
    validation = models.FloatField(default=0, null=True)
    metric = models.CharField(max_length=256, default=None, null=True, blank=True)

    model = models.CharField(max_length=256, )
    transforms = models.CharField(max_length=256, null=True)
    # pipeline = models.FileField(blank=True, upload_to='model/')
    pipeline = models.ForeignKey('Document', on_delete=models.SET_NULL, null=True, blank=True, related_name='pipeline')
    run_db = models.BooleanField(default=False)

    @property
    def tables(self):
        """Return table in DBMS"""
        if self.is_db:
            return get_tables(self.db_url)
        return []

    @property
    def columns(self):
        """Return columns in selected dataset"""
        if self.is_db:
            ds = get_table(self.db_url, self.table)
        else:
            ds = pd.read_csv(self.dataset.file)
        return ds.columns.to_list()


class ResultScenario(models.Model):
    """Result Scenario object to keep result of executed scenario"""
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE)

    execution_time = models.FloatField()
    throughput = models.FloatField()
    score = models.FloatField(null=True, blank=True)

    file_result = models.CharField(max_length=256, blank=True, null=True)
    query = models.CharField(max_length=1024, blank=True, null=True)


class Document(models.Model):
    """Document object"""
    filename = models.CharField(max_length=256)
    file = models.FileField()

    def __str__(self):
        return self.filename
