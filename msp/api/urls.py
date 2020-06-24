from django.urls import path, re_path

from .views import (
    MLManagerList,
    ScenarioList,
    ScenarioDetail,
    ScenarioTrainDetail,
    ScenarioTestDetail,
    ResultScenarioDetail,
    ScenarioFastDetail,
    ScenarioCompleteDetail,
    DocumentDetail,
    PipelineDetail,
    DBMSDetail,
)

app_name = 'msp'
urlpatterns = [
    # Url for get ML Manager components
    path('mlmanager/', MLManagerList.as_view()),  # [get, ]

    # Url for checking dbms detail
    path('dbms/', DBMSDetail.as_view()),  # [get, ]

    # Url for upload and download file
    path('document/<filename>/', DocumentDetail.as_view()),  # [get, post]

    # Url for checking pipeline object file
    path('pipeline/<filename>/', PipelineDetail.as_view()),  # [get]

    # Url for manage Scenario Model
    path('scenario/', ScenarioDetail.as_view()),  # [get, delete]
    path('scenario/train/', ScenarioTrainDetail.as_view()),  # [post]
    path('scenario/test/', ScenarioTestDetail.as_view()),  # [post]
    path('scenario/list/', ScenarioList.as_view()),  # [get]
    path('scenario/result/', ResultScenarioDetail.as_view()),  # [get]
    path('scenario/complete/', ScenarioCompleteDetail.as_view()),  # [get]

    # Url for execute simulation
    path('scenario/fast/', ScenarioFastDetail.as_view()),  # [post]
]
