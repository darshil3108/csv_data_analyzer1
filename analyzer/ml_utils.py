import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# Try to import XGBoost, but don't fail if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def get_algorithm_details(algorithm_name):
    """Get algorithm class and whether it's a classifier with speed optimizations"""
    regression_algorithms = {
        'linear_regression': (LinearRegression(), 'Linear Regression'),
        'ridge': (Ridge(), 'Ridge Regression'),
        'lasso': (Lasso(), 'Lasso Regression'),
        'polynomial': (Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]), 'Polynomial Regression'),
        'svr': (SVR(C=1.0, gamma='scale'), 'Support Vector Regression'),  # Optimized parameters
        'decision_tree_regressor': (DecisionTreeRegressor(max_depth=10), 'Decision Tree Regressor'),  # Limited depth
        'random_forest_regressor': (RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1), 'Random Forest Regressor'),  # Reduced trees, parallel processing
        'gradient_boosting_regressor': (GradientBoostingRegressor(n_estimators=20, max_depth=5), 'Gradient Boosting Regressor'),  # Reduced complexity
    }
    
    classification_algorithms = {
        'logistic_regression': (LogisticRegression(max_iter=500), 'Logistic Regression'),  # Reduced iterations
        'svc': (SVC(probability=True, C=1.0, gamma='scale'), 'Support Vector Classification'),  # Optimized parameters
        'decision_tree_classifier': (DecisionTreeClassifier(max_depth=10), 'Decision Tree Classifier'),  # Limited depth
        'random_forest_classifier': (RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1), 'Random Forest Classifier'),  # Reduced trees, parallel processing
        'knn': (KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors'),  # Reduced neighbors
        'gradient_boosting_classifier': (GradientBoostingClassifier(n_estimators=20, max_depth=5), 'Gradient Boosting Classifier'),  # Reduced complexity
        'ada_boost_classifier': (AdaBoostClassifier(n_estimators=20), 'AdaBoost Classifier'),  # Reduced estimators
    }
    
    # Add XGBoost if available with optimized parameters
    if XGBOOST_AVAILABLE:
        classification_algorithms['xgboost_classifier'] = (xgb.XGBClassifier(n_estimators=20, max_depth=5, n_jobs=-1), 'XGBoost Classifier')
        regression_algorithms['xgboost_regressor'] = (xgb.XGBRegressor(n_estimators=20, max_depth=5, n_jobs=-1), 'XGBoost Regressor')
    
    if algorithm_name in regression_algorithms:
        return regression_algorithms[algorithm_name][0], regression_algorithms[algorithm_name][1], False
    elif algorithm_name in classification_algorithms:
        return classification_algorithms[algorithm_name][0], classification_algorithms[algorithm_name][1], True
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

def prepare_data(df, x_columns, y_column, test_size=0.2):
    """Prepare data for ML model training with speed optimizations"""
    # Check if target is categorical (for classification)
    is_classification = False
    if df[y_column].dtype == 'object' or df[y_column].nunique() < 10:
        is_classification = True
    
    # Split features and target
    X = df[x_columns].copy()
    y = df[y_column]
    
    # Optimize categorical handling - limit categories for speed
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert to string and handle NaN
            X[col] = X[col].astype(str)
            X[col] = X[col].replace('nan', np.nan)
            
            # Limit categories to top 10 to prevent memory issues and improve speed
            value_counts = X[col].value_counts()
            if len(value_counts) > 10:
                top_categories = value_counts.head(10).index.tolist()
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
    
    # Identify numeric and categorical columns after type conversion
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'Int64', 'Float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create optimized preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Optimized categorical transformer with reduced max_categories
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))  # Reduced from 20 to 10
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Split data with custom test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, 
        stratify=y if is_classification and len(y.unique()) > 1 else None
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, is_classification

def train_and_evaluate_model(df, x_columns, y_column, algorithm_name, test_size=0.2):
    """Train a model and evaluate its performance with speed optimizations"""
    try:
        # Get algorithm with optimized parameters for speed
        model, algorithm_display_name, is_classifier_from_algo = get_algorithm_details(algorithm_name)
        
        # Prepare data with custom test size
        X_train, X_test, y_train, y_test, preprocessor, is_classification = prepare_data(
            df, x_columns, y_column, test_size
        )
        
        # Use algorithm's classification status if it's explicitly a classifier
        is_classifier = is_classifier_from_algo or is_classification
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model with speed optimization
        print(f"Training {algorithm_display_name}...")
        pipeline.fit(X_train, y_train)
        print("Training completed")
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if is_classifier:
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            metrics = {
                'r2_score': float(r2_score(y_test, y_pred)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }
        
        return pipeline, metrics, is_classifier, algorithm_display_name
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise e

def save_model(model, file_path):
    """Save model to file with error handling"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise e

def make_prediction(model, input_data, x_columns):
    """Make prediction using trained model with speed optimizations"""
    try:
        print(f"Making prediction with input: {input_data}")
        
        # Convert input data to DataFrame with proper column order
        input_df = pd.DataFrame([input_data], columns=x_columns)
        
        # Handle data types consistently with training (optimized)
        for col in input_df.columns:
            if col in input_data:
                value = input_data[col]
                
                # Optimized type conversion
                if isinstance(value, str):
                    clean_value = value.strip().replace(',', '').replace('$', '').replace('%', '')
                    try:
                        numeric_value = float(clean_value)
                        input_df[col] = numeric_value
                    except ValueError:
                        input_df[col] = str(value)
                else:
                    input_df[col] = value
        
        # Make prediction (this is already fast)
        prediction = model.predict(input_df)
        
        # Return first prediction
        result = prediction[0]
        
        # Convert numpy types to Python types for JSON serialization
        if isinstance(result, (np.int64, np.float64, np.int32, np.float32)):
            result = float(result)
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e

def validate_model_input(input_data, x_columns):
    """Validate input data before making predictions"""
    errors = []
    
    # Check if all required columns are present
    for col in x_columns:
        if col not in input_data:
            errors.append(f"Missing value for column '{col}'")
        elif input_data[col] is None or str(input_data[col]).strip() == '':
            errors.append(f"Empty value for column '{col}'")
    
    return errors
