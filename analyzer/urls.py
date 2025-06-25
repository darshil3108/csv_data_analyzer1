from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('column-selection/<int:csv_id>/', views.column_selection, name='column_selection'),
    path('data-cleaning/<int:csv_id>/', views.data_cleaning, name='data_cleaning'),
    path('analysis/<int:csv_id>/', views.analysis, name='analysis'),
    path('predict-page/<int:csv_id>/', views.predict_page, name='predict_page'),
    path('generate-chart/<int:csv_id>/', views.generate_chart_view, name='generate_chart'),
    path('get-chart-suggestions/<int:csv_id>/', views.get_chart_suggestions_view, name='get_chart_suggestions'),
    path('train-model/<int:csv_id>/', views.train_model_view, name='train_model'),
    path('predict/<int:model_id>/', views.predict_view, name='predict'),
]
