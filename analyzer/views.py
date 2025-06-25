from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from .models import CSVFile, Analysis, MLModel
from .forms import CSVUploadForm
from .utils import (
    read_csv_file, get_column_info, clean_data, 
    auto_clean_data, generate_chart, get_chart_suggestions, get_cleaning_options_for_column,
    detect_numeric_conversion_enhanced, calculate_statistics_safe
)
from .ml_utils import train_and_evaluate_model, save_model, make_prediction, validate_model_input
import json
import pandas as pd
import os
import uuid
import numpy as np
import time
import traceback

def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.save()
            
            # Read and analyze the CSV file
            df, error = read_csv_file(csv_file.file.path)
            if error:
                messages.error(request, f"Error reading CSV file: {error}")
                csv_file.delete()
                return render(request, 'upload.html', {'form': form})
            
            # Store column information
            column_info = get_column_info(df)
            csv_file.columns = column_info
            csv_file.save()
            
            return redirect('column_selection', csv_id=csv_file.id)
    else:
        form = CSVUploadForm()
    
    return render(request, 'upload.html', {'form': form})

def column_selection(request, csv_id):
    csv_file = get_object_or_404(CSVFile, id=csv_id)
    
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        csv_file.selected_columns = selected_columns
        csv_file.save()
        return redirect('data_cleaning', csv_id=csv_file.id)
    
    return render(request, 'column_selection.html', {
        'csv_file': csv_file,
        'columns': csv_file.columns
    })

def data_cleaning(request, csv_id):
    csv_file = get_object_or_404(CSVFile, id=csv_id)
    
    if request.method == 'POST':
        cleaning_method = request.POST.get('cleaning_method')
        
        if cleaning_method == 'auto':
            # Auto clean the data with enhanced type detection
            df, error = read_csv_file(csv_file.file.path)
            if error:
                messages.error(request, f"Error reading file: {error}")
                return redirect('upload_csv')
            
            # Filter selected columns
            if csv_file.selected_columns:
                df = df[csv_file.selected_columns]
            
            cleaned_df, cleaning_log = auto_clean_data(df)
            csv_file.cleaning_options = {'method': 'auto', 'log': cleaning_log}
            csv_file.save()
            
            # Show detailed summary of auto cleaning with enhanced information
            summary_messages = []
            type_conversions = []
            format_detections = []
            
            for col, info in cleaning_log.items():
                if info['type_converted']:
                    type_conversions.append(
                        f"'{col}': {info['original_dtype']} → {info['final_dtype']}"
                    )
                    
                    # Extract format information from method_used
                    method = info.get('method_used', '')
                    if 'detected' in method:
                        format_info = method.split('detected ')[1].split(' format')[0]
                        format_detections.append(f"'{col}': {format_info}")
                
                if info['action_taken'] != 'none':
                    summary_messages.append(
                        f"'{col}': {info['method_used']}"
                    )
            
            if type_conversions:
                messages.success(request, f"Enhanced type conversions: {'; '.join(type_conversions)}")
            
            if format_detections:
                messages.info(request, f"Detected formats: {'; '.join(format_detections)}")
            
            if summary_messages:
                messages.success(request, f"Data cleaning completed: {'; '.join(summary_messages)}")
            else:
                messages.info(request, "No cleaning was needed - all columns are already clean!")
            
        else:  # manual cleaning
            cleaning_options = {}
            for column in csv_file.selected_columns:
                missing_value = request.POST.get(f'missing_value_{column}')
                type_conversion = request.POST.get(f'type_conversion_{column}')
                column_action = request.POST.get(f'column_action_{column}')
                custom_value = request.POST.get(f'custom_value_{column}', '')
                
                cleaning_options[column] = {
                    'missing_value': missing_value,
                    'type_conversion': type_conversion,
                    'column_action': column_action,
                    'custom_value': custom_value
                }
            
            csv_file.cleaning_options = {'method': 'manual', 'options': cleaning_options}
            csv_file.save()
            
            messages.success(request, "Manual cleaning options saved successfully!")
        
        return redirect('analysis', csv_id=csv_file.id)
    
    # Get data preview for manual cleaning
    df, error = read_csv_file(csv_file.file.path)
    if error:
        messages.error(request, f"Error reading file: {error}")
        return redirect('upload_csv')
    
    if csv_file.selected_columns:
        df = df[csv_file.selected_columns]
    
    # Get detailed column info with enhanced cleaning options and type detection
    column_info = {}
    for col in df.columns:
        col_data = df[col]
        cleaning_options = get_cleaning_options_for_column(df, col)
        
        # Calculate basic statistics
        stats = {
            'count': len(col_data),
            'null_count': int(col_data.isnull().sum()),
            'null_percentage': round((col_data.isnull().sum() / len(col_data)) * 100, 1),
            'unique_count': int(col_data.nunique()),
            'dtype': str(col_data.dtype)
        }
        
        # Enhanced type conversion detection for object columns
        type_detection = {'can_convert': False}
        if col_data.dtype == 'object':
            can_convert, suggested_type, conversion_info = detect_numeric_conversion_enhanced(col_data)
            type_detection = {
                'can_convert': can_convert,
                'suggested_type': suggested_type,
                'conversion_info': conversion_info
            }
            # Calculate conversion percentage
            if can_convert and 'conversion_rate' in conversion_info:
                type_detection['conversion_percentage'] = round(conversion_info['conversion_rate'] * 100, 0)
                
            # Add format type information
            if 'format_types' in conversion_info:
                type_detection['format_types'] = conversion_info['format_types']
                type_detection['most_common_format'] = conversion_info.get('most_common_format', 'unknown')
        
        # Enhanced statistics calculation using safe numeric extraction
        if col_data.dtype in ['int64', 'float64', 'Int64', 'Float64']:
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                stats.update({
                    'mean': round(non_null_data.mean(), 2),
                    'median': round(non_null_data.median(), 2),
                    'std': round(non_null_data.std(), 2),
                    'min': non_null_data.min(),
                    'max': non_null_data.max()
                })
                mode_vals = non_null_data.mode()
                if not mode_vals.empty:
                    stats['mode'] = mode_vals.iloc[0]
        elif col_data.dtype == 'object':
            # Use enhanced numeric statistics for object columns
            numeric_stats = calculate_statistics_safe(col_data)
            if numeric_stats['count'] > 0 and numeric_stats['error'] is None:
                stats.update({
                    'numeric_mean': round(numeric_stats['mean'], 2),
                    'numeric_median': round(numeric_stats['median'], 2),
                    'numeric_count': numeric_stats['count']
                })
                if numeric_stats['mode'] is not None:
                    stats['numeric_mode'] = numeric_stats['mode']
            
            # Regular categorical statistics
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                mode_vals = non_null_data.mode()
                if not mode_vals.empty:
                    stats['mode'] = mode_vals.iloc[0]
                # Most common values
                value_counts = non_null_data.value_counts().head(3)
                stats['top_values'] = value_counts.to_dict()
        
        column_info[col] = {
            'stats': stats,
            'cleaning_options': cleaning_options,
            'type_detection': type_detection,
            'sample_values': col_data.dropna().head(5).tolist()
        }
    
    return render(request, 'data_cleaning.html', {
        'csv_file': csv_file,
        'column_info': column_info
    })

def analysis(request, csv_id):
    csv_file = get_object_or_404(CSVFile, id=csv_id)
    
    # Load and clean the data
    df, error = read_csv_file(csv_file.file.path)
    if error:
        messages.error(request, f"Error reading file: {error}")
        return redirect('upload_csv')
    
    # Filter selected columns
    if csv_file.selected_columns:
        df = df[csv_file.selected_columns]
    
    # Apply cleaning
    if csv_file.cleaning_options:
        if csv_file.cleaning_options.get('method') == 'auto':
            df, _ = auto_clean_data(df)
        else:
            cleaning_options = csv_file.cleaning_options.get('options', {})
            df = clean_data(df, cleaning_options)
    
    # Get available columns for analysis (updated to include new nullable integer types)
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    all_columns = df.columns.tolist()
    
    return render(request, 'analysis.html', {
        'csv_file': csv_file,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'all_columns': all_columns,
        'data_preview': df.head(10).to_html(classes='table table-striped', table_id='data-preview')
    })

def predict_page(request, csv_id):
    """Dedicated prediction page"""
    csv_file = get_object_or_404(CSVFile, id=csv_id)
    
    # Load and clean the data
    df, error = read_csv_file(csv_file.file.path)
    if error:
        messages.error(request, f"Error reading file: {error}")
        return redirect('upload_csv')
    
    # Filter selected columns
    if csv_file.selected_columns:
        df = df[csv_file.selected_columns]
    
    # Apply cleaning
    if csv_file.cleaning_options:
        if csv_file.cleaning_options.get('method') == 'auto':
            df, _ = auto_clean_data(df)
        else:
            cleaning_options = csv_file.cleaning_options.get('options', {})
            df = clean_data(df, cleaning_options)
    
    # Get available columns for analysis
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    all_columns = df.columns.tolist()
    
    return render(request, 'predict.html', {
        'csv_file': csv_file,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'all_columns': all_columns,
    })

def generate_chart_view(request, csv_id):
    if request.method == 'POST':
        csv_file = get_object_or_404(CSVFile, id=csv_id)
        
        chart_type = request.POST.get('chart_type')
        x_column = request.POST.get('x_column')
        y_column = request.POST.get('y_column') if request.POST.get('y_column') else None
        
        # Validate inputs
        if not chart_type:
            return JsonResponse({'error': 'Please select a chart type.'})
        
        if not x_column:
            return JsonResponse({'error': 'Please select an X-axis column.'})
        
        # Load and process data
        df, error = read_csv_file(csv_file.file.path)
        if error:
            return JsonResponse({'error': f"Error reading file: {error}"})
        
        # Filter and clean data
        if csv_file.selected_columns:
            # Check if selected columns still exist
            missing_columns = [col for col in csv_file.selected_columns if col not in df.columns]
            if missing_columns:
                return JsonResponse({'error': f"Selected columns not found: {', '.join(missing_columns)}"})
            
            df = df[csv_file.selected_columns]
        
        if csv_file.cleaning_options:
            try:
                if csv_file.cleaning_options.get('method') == 'auto':
                    df, _ = auto_clean_data(df)
                else:
                    cleaning_options = csv_file.cleaning_options.get('options', {})
                    df = clean_data(df, cleaning_options)
            except Exception as e:
                return JsonResponse({'error': f"Error cleaning data: {str(e)}"})
        
        # Check if dataframe is empty after cleaning
        if df.empty:
            return JsonResponse({'error': 'No data available after cleaning. Please check your data and cleaning options.'})
        
        # Generate chart
        buffer, result = generate_chart(df, chart_type, x_column, y_column)
        
        if buffer is None:
            return JsonResponse({'error': result})  # result contains error message
        
        try:
            # Save analysis record
            analysis = Analysis.objects.create(
                csv_file=csv_file,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column or ''
            )
            
            # Save chart image
            image_file = ContentFile(buffer.getvalue())
            analysis.chart_image.save(
                f'chart_{analysis.id}.png',
                image_file,
                save=True
            )
            
            return JsonResponse({
                'success': True,
                'image_base64': result,  # result contains base64 image
                'analysis_id': analysis.id,
                'message': f'{chart_type.title()} chart generated successfully!'
            })
            
        except Exception as e:
            return JsonResponse({'error': f"Error saving chart: {str(e)}"})
    
    return JsonResponse({'error': 'Invalid request method'})

def get_chart_suggestions_view(request, csv_id):
    if request.method == 'POST':
        csv_file = get_object_or_404(CSVFile, id=csv_id)
        x_column = request.POST.get('x_column')
        y_column = request.POST.get('y_column') if request.POST.get('y_column') else None
        
        # Load data
        df, error = read_csv_file(csv_file.file.path)
        if error:
            return JsonResponse({'error': f"Error reading file: {error}"})
        
        # Filter selected columns
        if csv_file.selected_columns:
            df = df[csv_file.selected_columns]
        
        suggestions = get_chart_suggestions(df, x_column, y_column)
        
        return JsonResponse({'suggestions': suggestions})
    
    return JsonResponse({'error': 'Invalid request method'})

def train_model_view(request, csv_id):
    """Train a machine learning model on the CSV data with improved error handling"""
    if request.method == 'POST':
        try:
            start_time = time.time()
            csv_file = get_object_or_404(CSVFile, id=csv_id)
            
            # Get parameters from request
            y_column = request.POST.get('y_column')
            algorithm = request.POST.get('algorithm')
            x_columns = request.POST.getlist('x_columns')
            test_size = float(request.POST.get('test_size', 0.2))
            
            print(f"Training request: y_column={y_column}, algorithm={algorithm}, x_columns={x_columns}")
            
            # Validate inputs
            if not y_column or not algorithm or not x_columns:
                return JsonResponse({
                    'success': False,
                    'error': 'Missing required parameters'
                })
            
            # Load and clean data
            df, error = read_csv_file(csv_file.file.path)
            if error:
                return JsonResponse({
                    'success': False,
                    'error': f"Error reading file: {error}"
                })
            
            # Filter selected columns
            if csv_file.selected_columns:
                df = df[csv_file.selected_columns]
            
            # Apply cleaning
            if csv_file.cleaning_options:
                if csv_file.cleaning_options.get('method') == 'auto':
                    df, _ = auto_clean_data(df)
                else:
                    cleaning_options = csv_file.cleaning_options.get('options', {})
                    df = clean_data(df, cleaning_options)
            
            # Check if all required columns exist
            missing_columns = [col for col in x_columns + [y_column] if col not in df.columns]
            if missing_columns:
                return JsonResponse({
                    'success': False,
                    'error': f"Columns not found: {', '.join(missing_columns)}"
                })
            
            # Remove rows where target column is null
            initial_rows = len(df)
            df = df.dropna(subset=[y_column])
            if len(df) == 0:
                return JsonResponse({
                    'success': False,
                    'error': f"No valid data remaining after removing null values from target column '{y_column}'"
                })
            
            # Check if we have enough data
            if len(df) < 10:
                return JsonResponse({
                    'success': False,
                    'error': f"Insufficient data for training. Only {len(df)} valid rows available. Need at least 10 rows."
                })
            
            # Remove rows where all feature columns are null
            df_features = df[x_columns + [y_column]].dropna()
            if len(df_features) < 5:
                return JsonResponse({
                    'success': False,
                    'error': f"Insufficient complete data for training. Only {len(df_features)} complete rows available."
                })
            
            print(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
            
            # Train model with the cleaned data and custom test size
            training_start = time.time()
            pipeline, metrics, is_classifier, algorithm_name = train_and_evaluate_model(
                df_features, x_columns, y_column, algorithm, test_size
            )
            print(f"Model training completed in {time.time() - training_start:.2f} seconds")
            
            # Save model to file
            model_filename = f"model_{uuid.uuid4().hex}.pkl"
            model_path = os.path.join('media', 'ml_models', model_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_model(pipeline, model_path)
            
            # Create MLModel instance
            ml_model = MLModel.objects.create(
                csv_file=csv_file,
                algorithm=algorithm,
                y_column=y_column,
                x_columns=x_columns,
                model_file=f'ml_models/{model_filename}',
                is_classifier=is_classifier,
                metrics=metrics
            )
            
            total_time = time.time() - start_time
            print(f"Total training process completed in {total_time:.2f} seconds")
            
            return JsonResponse({
                'success': True,
                'model_id': ml_model.id,
                'algorithm': algorithm,
                'algorithm_name': algorithm_name,
                'y_column': y_column,
                'x_columns': x_columns,
                'is_classifier': is_classifier,
                'metrics': metrics,
                'data_info': {
                    'total_rows': initial_rows,
                    'training_rows': len(df_features),
                    'dropped_rows': initial_rows - len(df_features),
                    'test_size': test_size
                },
                'training_time': round(total_time, 2)
            })
            
        except Exception as e:
            print(f"Error in train_model_view: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                'success': False,
                'error': f"Error training model: {str(e)}"
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def predict_view(request, model_id):
    """Make predictions using a trained model with improved error handling and speed"""
    if request.method == 'POST':
        try:
            start_time = time.time()
            print(f"Prediction request started for model {model_id}")
            
            ml_model = get_object_or_404(MLModel, id=model_id)
            print(f"Model loaded: {ml_model.algorithm}")
            
            # Parse input data
            input_data = json.loads(request.body)
            print(f"Input data received: {input_data}")
            
            # Validate input data
            validation_errors = validate_model_input(input_data, ml_model.x_columns)
            if validation_errors:
                return JsonResponse({
                    'success': False,
                    'error': f"Input validation failed: {'; '.join(validation_errors)}"
                })
            
            # Load model with timeout
            load_start = time.time()
            try:
                model = ml_model.load_model()
                print(f"Model loaded from file in {time.time() - load_start:.2f} seconds")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': f"Error loading model: {str(e)}"
                })
            
            # Make prediction with timeout
            prediction_start = time.time()
            try:
                prediction = make_prediction(model, input_data, ml_model.x_columns)
                print(f"Prediction completed in {time.time() - prediction_start:.2f} seconds")
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                print(traceback.format_exc())
                return JsonResponse({
                    'success': False,
                    'error': f"Error making prediction: {str(e)}"
                })
            
            # Format prediction based on type
            if isinstance(prediction, (np.int64, np.float64, np.int32, np.float32)):
                prediction = float(prediction)
            
            total_time = time.time() - start_time
            print(f"Total prediction process completed in {total_time:.2f} seconds")
            
            return JsonResponse({
                'success': True,
                'prediction': prediction,
                'prediction_time': round(total_time, 3)
            })
            
        except Exception as e:
            print(f"Unexpected error in predict_view: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def home(request):
    recent_files = CSVFile.objects.all().order_by('-uploaded_at')[:5]
    return render(request, 'home.html', {'recent_files': recent_files})
