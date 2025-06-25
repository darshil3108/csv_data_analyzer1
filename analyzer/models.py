from django.db import models
import os
import pickle

class CSVFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField(default=list, blank=True)
    selected_columns = models.JSONField(default=list, blank=True)
    cleaning_options = models.JSONField(default=dict, blank=True)
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        # Delete the file when the model instance is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

class Analysis(models.Model):
    csv_file = models.ForeignKey(CSVFile, on_delete=models.CASCADE)
    chart_type = models.CharField(max_length=50)
    x_column = models.CharField(max_length=255)
    y_column = models.CharField(max_length=255, blank=True, null=True)
    chart_image = models.ImageField(upload_to='charts/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.chart_type} - {self.x_column} vs {self.y_column}"

class MLModel(models.Model):
    csv_file = models.ForeignKey(CSVFile, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=100)
    y_column = models.CharField(max_length=255)
    x_columns = models.JSONField(default=list)
    model_file = models.FileField(upload_to='ml_models/')
    is_classifier = models.BooleanField(default=False)
    metrics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.algorithm} for {self.csv_file.name}"
    
    def delete(self, *args, **kwargs):
        # Delete the model file when the model instance is deleted
        if self.model_file:
            if os.path.isfile(self.model_file.path):
                os.remove(self.model_file.path)
        super().delete(*args, **kwargs)
    
    def load_model(self):
        """Load the pickled model from file"""
        with open(self.model_file.path, 'rb') as f:
            return pickle.load(f)
