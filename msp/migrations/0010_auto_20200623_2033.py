# Generated by Django 3.0.2 on 2020-06-23 18:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('msp', '0009_auto_20200311_1027'),
    ]

    operations = [
        migrations.AlterField(
            model_name='resultscenario',
            name='query',
            field=models.CharField(blank=True, max_length=1024, null=True),
        ),
    ]
