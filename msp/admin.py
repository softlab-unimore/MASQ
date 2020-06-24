from django.contrib import admin

from .models import Scenario, ResultScenario, Document

admin.site.register(Scenario)
admin.site.register(ResultScenario)
admin.site.register(Document)

