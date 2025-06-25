import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.core.files.base import ContentFile
import os
import re


def read_csv_file(file_path):
    """Read CSV file and return DataFrame with basic info"""
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

def get_column_info(df):
    """Get detailed information about each column"""
    column_info = {}
    for col in df.columns:
        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': df[col].dropna().head(5).tolist()
        }
        column_info[col] = info
    return column_info


def generate_chart(df, chart_type, x_column, y_column=None):
    """Generate chart with enhanced flexibility for categorical data"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # Validate data exists
        if df.empty:
            plt.close()
            return None, "No data available to generate chart."
        
        # Check if columns exist
        if x_column not in df.columns:
            plt.close()
            return None, f"Column '{x_column}' not found in the dataset."
        
        if y_column and y_column not in df.columns:
            plt.close()
            return None, f"Column '{y_column}' not found in the dataset."
        
        # Remove rows where x_column is null
        df_clean = df.dropna(subset=[x_column])
        if df_clean.empty:
            plt.close()
            return None, f"No valid data found in column '{x_column}'. All values are null."
        
        if chart_type == 'histogram':
            # Histogram logic remains the same
            if df_clean[x_column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                numeric_data = df_clean[x_column].dropna()
                if len(numeric_data) == 0:
                    plt.close()
                    return None, f"No numeric data found in column '{x_column}'."
                
                n, bins, patches = ax.hist(numeric_data, bins=min(30, len(numeric_data.unique())), 
                                         alpha=0.7, color='skyblue', edgecolor='black', 
                                         label=f'{x_column} Distribution')
                
                mean_val = numeric_data.mean()
                median_val = numeric_data.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                
                ax.set_xlabel(x_column)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {x_column}')
                ax.legend(loc='upper right')
                
            else:
                # For categorical data
                value_counts = df_clean[x_column].value_counts().head(20)
                if len(value_counts) == 0:
                    plt.close()
                    return None, f"No data to display for column '{x_column}'."
                
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                             color='skyblue', label=f'{x_column} Counts')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_xlabel(x_column)
                ax.set_ylabel('Count')
                ax.set_title(f'Count Plot of {x_column}')
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
                
                ax.legend(loc='upper right')
        
        elif chart_type == 'bar':
            # Bar chart logic remains the same
            if not y_column:
                plt.close()
                return None, "Bar chart requires both X and Y columns. Please select a Y column."
            
            df_clean = df.dropna(subset=[x_column, y_column])
            if df_clean.empty:
                plt.close()
                return None, f"No valid data found for columns '{x_column}' and '{y_column}'."
            
            if df_clean[x_column].dtype == 'object' and df_clean[y_column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                grouped = df_clean.groupby(x_column)[y_column].agg(['mean', 'count']).head(20)
                if len(grouped) == 0:
                    plt.close()
                    return None, f"No data to group by '{x_column}'."
                
                bars = ax.bar(range(len(grouped)), grouped['mean'].values, 
                             color='lightcoral', alpha=0.8, 
                             label=f'Average {y_column}')
                
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                ax.set_xlabel(x_column)
                ax.set_ylabel(f'Average {y_column}')
                ax.set_title(f'Average {y_column} by {x_column}')
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    count = grouped['count'].iloc[i]
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}\n(n={count})', ha='center', va='bottom', fontsize=8)
                
                total_records = grouped['count'].sum()
                ax.legend([f'Average {y_column}', f'Total Records: {total_records}'], 
                         loc='upper right')
                
            else:
                plt.close()
                return None, f"Bar chart requires categorical X-axis ('{x_column}') and numeric Y-axis ('{y_column}')."
        
        elif chart_type == 'line':
            # MODIFIED: Enhanced line chart with categorical support
            if not y_column:
                plt.close()
                return None, "Line chart requires both X and Y columns. Please select a Y column."
            
            df_clean = df.dropna(subset=[x_column, y_column])
            if df_clean.empty:
                plt.close()
                return None, f"No valid data found for columns '{x_column}' and '{y_column}'."
            
            x_is_numeric = df_clean[x_column].dtype in ['int64', 'float64', 'Int64', 'Float64']
            y_is_numeric = df_clean[y_column].dtype in ['int64', 'float64', 'Int64', 'Float64']
            
            if x_is_numeric and y_is_numeric:
                # Both numeric - traditional line chart
                if len(df_clean) < 2:
                    plt.close()
                    return None, "Line chart requires at least 2 data points."
                
                sorted_data = df_clean.sort_values(x_column)
                
                line = ax.plot(sorted_data[x_column], sorted_data[y_column], 
                              marker='o', linewidth=2, markersize=4, 
                              label=f'{y_column} vs {x_column}', color='blue')
                
                # Add trend line
                z = np.polyfit(sorted_data[x_column], sorted_data[y_column], 1)
                p = np.poly1d(z)
                ax.plot(sorted_data[x_column], p(sorted_data[x_column]), 
                       linestyle='--', color='red', alpha=0.8, 
                       label=f'Trend Line (slope: {z[0]:.3f})')
                
                correlation = sorted_data[x_column].corr(sorted_data[y_column])
                
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'{y_column} vs {x_column}')
                ax.legend([f'{y_column} vs {x_column}', 
                          f'Trend Line (slope: {z[0]:.3f})',
                          f'Correlation: {correlation:.3f}'], 
                         loc='upper left')
                
            elif not x_is_numeric and y_is_numeric:
                # Categorical X, Numeric Y - line chart with categorical axis
                grouped = df_clean.groupby(x_column)[y_column].agg(['mean', 'count', 'std']).reset_index()
                grouped = grouped.head(20)  # Limit to 20 categories
                
                if len(grouped) == 0:
                    plt.close()
                    return None, f"No data to group by '{x_column}'."
                
                x_positions = range(len(grouped))
                means = grouped['mean'].values
                stds = grouped['std'].fillna(0).values
                
                # Line plot with error bars
                line = ax.plot(x_positions, means, marker='o', linewidth=2, markersize=6, 
                              color='blue', label=f'Average {y_column}')
                
                # Add error bars if we have standard deviation
                if not all(stds == 0):
                    ax.errorbar(x_positions, means, yerr=stds, fmt='none', 
                               color='blue', alpha=0.5, capsize=3, label='±1 Std Dev')
                
                ax.set_xticks(x_positions)
                ax.set_xticklabels(grouped[x_column], rotation=45, ha='right')
                ax.set_xlabel(x_column)
                ax.set_ylabel(f'Average {y_column}')
                ax.set_title(f'Average {y_column} by {x_column} (Line Chart)')
                
                # Add value labels
                for i, (mean_val, count) in enumerate(zip(means, grouped['count'])):
                    ax.text(i, mean_val + max(means) * 0.02, f'{mean_val:.1f}\n(n={count})', 
                           ha='center', va='bottom', fontsize=8)
                
                legend_labels = [f'Average {y_column}']
                if not all(stds == 0):
                    legend_labels.append('±1 Std Dev')
                legend_labels.append(f'Total Records: {grouped["count"].sum()}')
                ax.legend(legend_labels, loc='upper left')
                
            elif x_is_numeric and not y_is_numeric:
                # Numeric X, Categorical Y - less common but possible
                plt.close()
                return None, f"Line chart with numeric X-axis and categorical Y-axis is not supported. Try swapping the axes or use a different chart type."
                
            else:
                # Both categorical
                plt.close()
                return None, f"Line chart requires at least one numeric column. Both '{x_column}' and '{y_column}' are categorical."
        
        elif chart_type == 'scatter':
            # MODIFIED: Enhanced scatter plot with categorical support
            if not y_column:
                plt.close()
                return None, "Scatter plot requires both X and Y columns. Please select a Y column."
            
            df_clean = df.dropna(subset=[x_column, y_column])
            if df_clean.empty:
                plt.close()
                return None, f"No valid data found for columns '{x_column}' and '{y_column}'."
            
            x_is_numeric = df_clean[x_column].dtype in ['int64', 'float64', 'Int64', 'Float64']
            y_is_numeric = df_clean[y_column].dtype in ['int64', 'float64', 'Int64', 'Float64']
            
            if x_is_numeric and y_is_numeric:
                # Both numeric - traditional scatter plot
                scatter = ax.scatter(df_clean[x_column], df_clean[y_column], 
                                   alpha=0.6, c='green', s=50, 
                                   label=f'{y_column} vs {x_column}')
                
                # Add trend line
                z = np.polyfit(df_clean[x_column], df_clean[y_column], 1)
                p = np.poly1d(z)
                ax.plot(df_clean[x_column], p(df_clean[x_column]), 
                       linestyle='--', color='red', linewidth=2, 
                       label=f'Trend Line')
                
                correlation = df_clean[x_column].corr(df_clean[y_column])
                
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'{y_column} vs {x_column} (Scatter Plot)')
                
                legend_labels = [
                    f'Data Points (n={len(df_clean)})',
                    f'Trend Line',
                    f'Correlation: {correlation:.3f}'
                ]
                ax.legend(legend_labels, loc='upper left')
                
            elif not x_is_numeric and y_is_numeric:
                # Categorical X, Numeric Y - strip plot / categorical scatter
                categories = df_clean[x_column].unique()
                categories = categories[:20]  # Limit to 20 categories
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                
                for i, category in enumerate(categories):
                    cat_data = df_clean[df_clean[x_column] == category]
                    if len(cat_data) > 0:
                        # Add some jitter to x-axis for better visibility
                        x_jitter = np.random.normal(i, 0.1, len(cat_data))
                        ax.scatter(x_jitter, cat_data[y_column], 
                                 alpha=0.6, c=[colors[i]], s=50, 
                                 label=f'{category} (n={len(cat_data)})')
                
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'{y_column} by {x_column} (Categorical Scatter)')
                
                # Add box plot overlay for better visualization
                box_data = [df_clean[df_clean[x_column] == cat][y_column].values 
                           for cat in categories]
                box_plot = ax.boxplot(box_data, positions=range(len(categories)), 
                                     widths=0.6, patch_artist=False, 
                                     boxprops=dict(alpha=0.3),
                                     showfliers=False)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            elif x_is_numeric and not y_is_numeric:
                # Numeric X, Categorical Y - horizontal categorical scatter
                categories = df_clean[y_column].unique()
                categories = categories[:20]  # Limit to 20 categories
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
                
                for i, category in enumerate(categories):
                    cat_data = df_clean[df_clean[y_column] == category]
                    if len(cat_data) > 0:
                        # Add some jitter to y-axis for better visibility
                        y_jitter = np.random.normal(i, 0.1, len(cat_data))
                        ax.scatter(cat_data[x_column], y_jitter, 
                                 alpha=0.6, c=[colors[i]], s=50, 
                                 label=f'{category} (n={len(cat_data)})')
                
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f'{x_column} by {y_column} (Categorical Scatter)')
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            else:
                # Both categorical - not ideal for scatter plot
                plt.close()
                return None, f"Scatter plot requires at least one numeric column. Both '{x_column}' and '{y_column}' are categorical. Consider using a bar chart instead."
        
        elif chart_type == 'box':
            # Box plot logic remains the same
            if df_clean[x_column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                numeric_data = df_clean[x_column].dropna()
                if len(numeric_data) == 0:
                    plt.close()
                    return None, f"No numeric data found in column '{x_column}'."
                
                if len(numeric_data) < 5:
                    plt.close()
                    return None, f"Box plot requires at least 5 data points. Found only {len(numeric_data)}."
                
                box_plot = ax.boxplot(numeric_data, patch_artist=True, 
                                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                                     medianprops=dict(color='red', linewidth=2))
                
                q1 = numeric_data.quantile(0.25)
                median = numeric_data.median()
                q3 = numeric_data.quantile(0.75)
                iqr = q3 - q1
                lower_whisker = max(numeric_data.min(), q1 - 1.5 * iqr)
                upper_whisker = min(numeric_data.max(), q3 + 1.5 * iqr)
                
                ax.set_ylabel(x_column)
                ax.set_title(f'Box Plot of {x_column}')
                ax.set_xticklabels([x_column])
                
                legend_elements = [
                    plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='Box (IQR)'),
                    plt.Line2D([0], [0], color='red', linewidth=2, label=f'Median: {median:.2f}'),
                    plt.Line2D([0], [0], color='black', linewidth=1, label=f'Q1: {q1:.2f}, Q3: {q3:.2f}'),
                    plt.Line2D([0], [0], color='black', linewidth=1, linestyle='--', 
                              label=f'Whiskers: {lower_whisker:.2f} - {upper_whisker:.2f}')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
            else:
                plt.close()
                return None, f"Box plot requires numeric data. Column '{x_column}' contains non-numeric data."
        
        elif chart_type == 'pie':
            # Pie chart logic remains the same
            if df_clean[x_column].dtype == 'object':
                value_counts = df_clean[x_column].value_counts().head(10)
                if len(value_counts) == 0:
                    plt.close()
                    return None, f"No data to display for column '{x_column}'."
                
                total = value_counts.sum()
                significant_counts = value_counts[value_counts / total >= 0.01]
                
                if len(significant_counts) == 0:
                    plt.close()
                    return None, f"All categories in '{x_column}' are too small to display in pie chart."
                
                others_sum = value_counts[value_counts / total < 0.01].sum()
                if others_sum > 0:
                    significant_counts['Others'] = others_sum
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(significant_counts)))
                
                wedges, texts, autotexts = ax.pie(significant_counts.values, 
                                                 labels=significant_counts.index,
                                                 autopct='%1.1f%%', 
                                                 startangle=90,
                                                 colors=colors,
                                                 explode=[0.05] * len(significant_counts))
                
                ax.set_title(f'Distribution of {x_column}')
                
                legend_labels = []
                for i, (category, count) in enumerate(significant_counts.items()):
                    percentage = (count / total) * 100
                    legend_labels.append(f'{category}: {count} ({percentage:.1f}%)')
                
                ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
                
            else:
                plt.close()
                return None, f"Pie chart requires categorical data. Column '{x_column}' contains numeric data."
        
        else:
            plt.close()
            return None, f"Unsupported chart type: '{chart_type}'."
        
        # Add grid for better readability (except for pie charts)
        if chart_type != 'pie':
            ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        if chart_type == 'pie' or (chart_type == 'scatter' and not x_is_numeric):
            plt.subplots_adjust(right=0.7)
        
        # Save plot to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64 for display
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        return buffer, image_base64
        
    except Exception as e:
        plt.close()
        return None, f"Error generating chart: {str(e)}"

def get_chart_suggestions(df, x_column, y_column=None):
    """MODIFIED: Suggest appropriate chart types with enhanced categorical support"""
    suggestions = []
    
    x_type = 'numeric' if df[x_column].dtype in ['int64', 'float64', 'Int64', 'Float64'] else 'categorical'
    
    if y_column:
        y_type = 'numeric' if df[y_column].dtype in ['int64', 'float64', 'Int64', 'Float64'] else 'categorical'
        
        if x_type == 'numeric' and y_type == 'numeric':
            # Both numeric - all chart types available
            suggestions = ['scatter', 'line']
        elif x_type == 'categorical' and y_type == 'numeric':
            # Categorical X, Numeric Y - bar, line, and scatter all work
            suggestions = ['bar', 'line', 'scatter']
        elif x_type == 'numeric' and y_type == 'categorical':
            # Numeric X, Categorical Y - scatter works, line doesn't
            suggestions = ['scatter', 'bar']  # Note: bar would need axes swapped
        else:
            # Both categorical - only bar chart makes sense
            suggestions = ['bar']
    else:
        # Single column
        if x_type == 'numeric':
            suggestions = ['histogram', 'box']
        else:
            suggestions = ['bar', 'pie']
    
    return suggestions

def validate_chart_data(df, chart_type, x_column, y_column=None):
    """Validate data before chart generation"""
    errors = []
    
    # Basic validations
    if df.empty:
        errors.append("Dataset is empty.")
        return errors
    
    if x_column not in df.columns:
        errors.append(f"Column '{x_column}' not found.")
        return errors
    
    if y_column and y_column not in df.columns:
        errors.append(f"Column '{y_column}' not found.")
        return errors
    
    # Chart-specific validations
    if chart_type in ['line', 'scatter', 'bar'] and not y_column:
        errors.append(f"{chart_type.title()} chart requires both X and Y columns.")
    
    # Data type validations
    x_data = df[x_column].dropna()
    if len(x_data) == 0:
        errors.append(f"Column '{x_column}' has no valid data.")
    
    if y_column:
        y_data = df[y_column].dropna()
        if len(y_data) == 0:
            errors.append(f"Column '{y_column}' has no valid data.")
    
    return errors

def extract_number_from_string_enhanced(value):
    """Enhanced extraction of numeric values from formatted strings"""
    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string and clean whitespace
    str_val = str(value).strip()
    
    # If already a number, return it
    try:
        return float(str_val)
    except ValueError:
        pass
    
    # Remove common currency symbols and spaces
    cleaned = re.sub(r'[$€£¥₹₽¢₩₪₦₨₱₡₴₸₼₾]', '', str_val)
    cleaned = re.sub(r'[,\s]', '', cleaned)  # Remove commas and spaces
    
    # Handle percentage
    if '%' in cleaned:
        number_part = re.sub(r'[^0-9.-]', '', cleaned)
        try:
            return float(number_part) / 100
        except ValueError:
            return np.nan
    
    # Handle suffixes (K, M, B, T for thousands, millions, billions, trillions)
    multipliers = {
        'K': 1000, 'k': 1000,
        'M': 1000000, 'm': 1000000, 'mil': 1000000, 'million': 1000000,
        'B': 1000000000, 'b': 1000000000, 'bil': 1000000000, 'billion': 1000000000,
        'T': 1000000000000, 't': 1000000000000, 'tril': 1000000000000, 'trillion': 1000000000000
    }
    
    # Check for suffix (case insensitive)
    multiplier = 1
    cleaned_lower = cleaned.lower()
    
    for suffix, mult in multipliers.items():
        if cleaned_lower.endswith(suffix.lower()):
            multiplier = mult
            # Remove the suffix
            cleaned = cleaned[:len(cleaned)-len(suffix)]
            break
    
    # Handle parentheses for negative numbers (accounting format)
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    # Extract number using regex (handles negative numbers and decimals)
    number_match = re.search(r'-?\d*\.?\d+', cleaned)
    if number_match:
        try:
            number = float(number_match.group())
            return number * multiplier
        except ValueError:
            return np.nan
    
    return np.nan

def detect_numeric_conversion_enhanced(series):
    """Enhanced detection with better format handling and validation - MODIFIED to only suggest float"""
    if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        return False, None, {}
    
    # Sample non-null values
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return False, None, {}
    
    # Test conversion on sample (use larger sample for better accuracy)
    sample_size = min(200, len(non_null_series))
    sample = non_null_series.sample(n=sample_size, random_state=42) if len(non_null_series) > sample_size else non_null_series
    
    converted_values = []
    original_values = []
    conversion_examples = []
    format_types = {
        'currency': 0,
        'percentage': 0, 
        'suffix_notation': 0,
        'plain_number': 0,
        'accounting': 0,
        'failed': 0
    }
    
    for value in sample:
        converted = extract_number_from_string_enhanced(value)
        converted_values.append(converted)
        original_values.append(value)
        
        # Categorize the format type
        str_val = str(value).strip()
        if not pd.isna(converted):
            if len(conversion_examples) < 8:  # Show more examples
                conversion_examples.append({
                    'original': str_val,
                    'converted': converted
                })
            
            # Detect format type
            if re.search(r'[$€£¥₹₽¢₩₪₦₨₱₡₴₸₼₾]', str_val):
                format_types['currency'] += 1
            elif '%' in str_val:
                format_types['percentage'] += 1
            elif re.search(r'[kmbtKMBT]$', str_val.replace(' ', '')):
                format_types['suffix_notation'] += 1
            elif str_val.startswith('(') and str_val.endswith(')'):
                format_types['accounting'] += 1
            else:
                try:
                    float(str_val)
                    format_types['plain_number'] += 1
                except:
                    format_types['suffix_notation'] += 1  # Likely has some formatting
        else:
            format_types['failed'] += 1
    
    # Calculate conversion success rate
    successful_conversions = sum(1 for val in converted_values if not pd.isna(val))
    conversion_rate = successful_conversions / len(converted_values)
    
    # More strict threshold for auto-conversion (85% success rate)
    can_convert = conversion_rate >= 0.85
    
    if not can_convert:
        return False, None, {
            'conversion_rate': conversion_rate,
            'sample_size': sample_size,
            'successful_conversions': successful_conversions,
            'format_types': format_types
        }
    
    # MODIFIED: Always suggest float64, never int64
    suggested_type = 'float64'
    
    # Determine most common format
    most_common_format = max(format_types, key=format_types.get)
    
    conversion_info = {
        'conversion_rate': conversion_rate,
        'sample_size': sample_size,
        'successful_conversions': successful_conversions,
        'suggested_type': suggested_type,
        'conversion_examples': conversion_examples,
        'format_types': format_types,
        'most_common_format': most_common_format,
        'sample_values': [str(val) for val in sample.head(5).tolist()]
    }
    
    return can_convert, suggested_type, conversion_info

def convert_column_type_enhanced(series, target_type):
    """Enhanced column type conversion with better format handling - MODIFIED to handle float-only conversion"""
    try:
        if target_type == 'int64':
            # MODIFIED: Convert to float first, then to nullable int to preserve precision
            if series.dtype == 'object':
                # Use enhanced extraction for object columns
                converted = series.apply(extract_number_from_string_enhanced)
                # Convert to float first, then round and convert to nullable int
                float_series = pd.to_numeric(converted, errors='coerce')
                return float_series.round().astype('Int64'), None
            else:
                return pd.to_numeric(series, errors='coerce').round().astype('Int64'), None
                
        elif target_type == 'float64':
            if series.dtype == 'object':
                # Use enhanced extraction for object columns
                converted = series.apply(extract_number_from_string_enhanced)
                return pd.to_numeric(converted, errors='coerce'), None
            else:
                return pd.to_numeric(series, errors='coerce'), None
                
        elif target_type == 'bool':
            # Enhanced boolean conversion
            def convert_to_bool(val):
                if pd.isna(val):
                    return np.nan
                str_val = str(val).lower().strip()
                if str_val in ['true', '1', 'yes', 'y', 'on', 't']:
                    return True
                elif str_val in ['false', '0', 'no', 'n', 'off', 'f']:
                    return False
                else:
                    # Try to convert to number first
                    try:
                        num_val = extract_number_from_string_enhanced(val)
                        if not pd.isna(num_val):
                            return bool(num_val)
                        return np.nan
                    except:
                        return np.nan
            
            return series.apply(convert_to_bool), None
            
        elif target_type == 'object':
            return series.astype(str), None
            
        else:
            return series, f"Unsupported target type: {target_type}"
            
    except Exception as e:
        return series, f"Conversion error: {str(e)}"

def get_numeric_only_data(series):
    """Extract only numeric data from a series, excluding non-convertible values"""
    if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        return series.dropna()
    
    # For object columns, try to extract numeric values
    numeric_values = series.apply(extract_number_from_string_enhanced)
    return numeric_values.dropna()

def calculate_statistics_safe(series):
    """Calculate statistics only on numeric data"""
    numeric_data = get_numeric_only_data(series)
    
    if len(numeric_data) == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'mode': None,
            'std': None,
            'min': None,
            'max': None,
            'error': 'No numeric data found'
        }
    
    try:
        stats = {
            'count': len(numeric_data),
            'mean': float(numeric_data.mean()),
            'median': float(numeric_data.median()),
            'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0,
            'min': float(numeric_data.min()),
            'max': float(numeric_data.max()),
            'error': None
        }
        
        # Calculate mode safely
        mode_vals = numeric_data.mode()
        if not mode_vals.empty:
            stats['mode'] = float(mode_vals.iloc[0])
        else:
            stats['mode'] = None
            
        return stats
    except Exception as e:
        return {
            'count': len(numeric_data),
            'mean': None,
            'median': None,
            'mode': None,
            'std': None,
            'min': None,
            'max': None,
            'error': f'Error calculating statistics: {str(e)}'
        }

def get_cleaning_options_for_column(df, column):
    """Get available cleaning options separated into missing value handling and type conversion - MODIFIED"""
    col_data = df[column]
    null_count = col_data.isnull().sum()
    current_dtype = str(col_data.dtype)
    
    # Missing value handling options (unchanged)
    missing_value_options = []
    
    if null_count == 0:
        missing_value_options.append({
            'value': 'keep', 
            'label': 'No missing values - keep as is', 
            'recommended': True
        })
    else:
        missing_value_options.append({
            'value': 'keep', 
            'label': 'Keep missing values as they are', 
            'recommended': False
        })
        missing_value_options.append({
            'value': 'drop_nulls', 
            'label': 'Remove rows with missing values', 
            'recommended': False
        })
        
        # Statistical filling options based on current data type or numeric conversion potential
        if col_data.dtype in ['int64', 'float64', 'Int64', 'Float64']:
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                mean_val = non_null_data.mean()
                median_val = non_null_data.median()
                missing_value_options.extend([
                    {
                        'value': 'fill_mean', 
                        'label': f'Fill with mean ({mean_val:.2f})', 
                        'recommended': True
                    },
                    {
                        'value': 'fill_median', 
                        'label': f'Fill with median ({median_val:.2f})', 
                        'recommended': True
                    }
                ])
                
                mode_vals = non_null_data.mode()
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    missing_value_options.append({
                        'value': 'fill_mode', 
                        'label': f'Fill with mode ({mode_val})', 
                        'recommended': False
                    })
            
            missing_value_options.append({
                'value': 'fill_zero', 
                'label': 'Fill with zero', 
                'recommended': False
            })
            
        elif col_data.dtype == 'object':
            # Check if object column can be converted to numeric for statistical filling
            numeric_stats = calculate_statistics_safe(col_data)
            
            if numeric_stats['count'] > 0 and numeric_stats['error'] is None:
                # Object column has numeric data - offer numeric statistical options
                missing_value_options.extend([
                    {
                        'value': 'fill_mean', 
                        'label': f'Fill with mean of numeric values ({numeric_stats["mean"]:.2f})', 
                        'recommended': True
                    },
                    {
                        'value': 'fill_median', 
                        'label': f'Fill with median of numeric values ({numeric_stats["median"]:.2f})', 
                        'recommended': True
                    }
                ])
                
                if numeric_stats['mode'] is not None:
                    missing_value_options.append({
                        'value': 'fill_mode', 
                        'label': f'Fill with mode of numeric values ({numeric_stats["mode"]})', 
                        'recommended': False
                    })
            
            # Regular categorical mode
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                mode_vals = non_null_data.mode()
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    missing_value_options.append({
                        'value': 'fill_mode', 
                        'label': f'Fill with most common value ("{mode_val}")', 
                        'recommended': True if numeric_stats['count'] == 0 else False
                    })
            
            missing_value_options.extend([
                {
                    'value': 'fill_unknown', 
                    'label': 'Fill with "Unknown"', 
                    'recommended': False
                },
                {
                    'value': 'fill_empty', 
                    'label': 'Fill with empty string', 
                    'recommended': False
                }
            ])
            
        elif col_data.dtype == 'bool':
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                mode_vals = non_null_data.mode()
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    missing_value_options.append({
                        'value': 'fill_mode', 
                        'label': f'Fill with mode ({mode_val})', 
                        'recommended': True
                    })
            
            missing_value_options.extend([
                {
                    'value': 'fill_false', 
                    'label': 'Fill with False', 
                    'recommended': False
                },
                {
                    'value': 'fill_true', 
                    'label': 'Fill with True', 
                    'recommended': False
                }
            ])
        
        # Custom value option
        missing_value_options.append({
            'value': 'fill_custom', 
            'label': 'Fill with custom value', 
            'recommended': False
        })
    
    # MODIFIED: Type conversion options - only suggest float for numeric conversion
    type_conversion_options = []
    
    # Always include "keep current type" option
    type_conversion_options.append({
        'value': 'keep_type', 
        'label': f'Keep as {current_dtype}', 
        'recommended': True
    })
    
    if col_data.dtype == 'object':
        # Check if can be converted to numeric using enhanced detection
        can_convert, suggested_type, conversion_info = detect_numeric_conversion_enhanced(col_data)
        
        if can_convert:
            conversion_rate = conversion_info.get('conversion_rate', 0) * 100
            format_info = ""
            if 'most_common_format' in conversion_info:
                format_info = f" - {conversion_info['most_common_format']} format detected"
            
            # MODIFIED: Always suggest float64, never int64
            type_conversion_options.append({
                'value': 'convert_to_float64',
                'label': f'Convert to decimal/float ({conversion_rate:.0f}% convertible{format_info})',
                'recommended': True,
                'conversion_info': conversion_info
            })
        
        # MODIFIED: Manual conversion options - only offer float, not int
        type_conversion_options.extend([
            {
                'value': 'convert_to_float64',
                'label': 'Convert to decimal/float (extracts numbers from formatted text)',
                'recommended': False
            },
            {
                'value': 'convert_to_bool',
                'label': 'Convert to boolean (true/false)',
                'recommended': False
            }
        ])
        
    elif col_data.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        # MODIFIED: Remove int conversion option for float columns
        type_conversion_options.extend([
            {
                'value': 'convert_to_object',
                'label': 'Convert to text/categorical',
                'recommended': False
            },
            {
                'value': 'convert_to_bool',
                'label': 'Convert to boolean',
                'recommended': False
            }
        ])
        
    elif col_data.dtype == 'bool':
        type_conversion_options.extend([
            {
                'value': 'convert_to_object',
                'label': 'Convert to text',
                'recommended': False
            },
            {
                'value': 'convert_to_float64',  # MODIFIED: Changed from int64 to float64
                'label': 'Convert to decimal/float (True=1.0, False=0.0)',
                'recommended': False
            }
        ])
    
    # Column removal option (separate category)
    column_actions = [
        {
            'value': 'keep_column', 
            'label': 'Keep this column', 
            'recommended': True
        },
        {
            'value': 'remove_column', 
            'label': 'Remove entire column', 
            'recommended': False
        }
    ]
    
    return {
        'missing_value_options': missing_value_options,
        'type_conversion_options': type_conversion_options,
        'column_actions': column_actions
    }

def clean_data(df, cleaning_options):
    """Apply manual cleaning operations with enhanced type conversion handling"""
    cleaned_df = df.copy()
    
    for column, options in cleaning_options.items():
        if column not in cleaned_df.columns:
            continue
        
        # Handle column removal first
        column_action = options.get('column_action', 'keep_column')
        if column_action == 'remove_column':
            cleaned_df = cleaned_df.drop(columns=[column])
            continue
        
        # Handle type conversion using enhanced functions
        type_action = options.get('type_conversion', 'keep_type')
        if type_action != 'keep_type':
            if type_action.startswith('convert_to_'):
                target_type = type_action.replace('convert_to_', '')
                converted_series, error = convert_column_type_enhanced(cleaned_df[column], target_type)
                if error is None:
                    cleaned_df[column] = converted_series
        
        # Handle missing values with enhanced numeric support
        missing_action = options.get('missing_value', 'keep')
        if missing_action == 'drop_nulls':
            cleaned_df = cleaned_df.dropna(subset=[column])
        elif missing_action == 'fill_mean':
            if cleaned_df[column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                mean_val = cleaned_df[column].mean()
                cleaned_df[column].fillna(mean_val, inplace=True)
            else:
                # For object columns, use numeric-only mean
                numeric_stats = calculate_statistics_safe(cleaned_df[column])
                if numeric_stats['mean'] is not None:
                    # Convert to numeric, fill mean, then convert back if needed
                    numeric_series = cleaned_df[column].apply(extract_number_from_string_enhanced)
                    mean_val = numeric_stats['mean']
                    cleaned_df[column] = cleaned_df[column].fillna(mean_val)
        elif missing_action == 'fill_median':
            if cleaned_df[column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                median_val = cleaned_df[column].median()
                cleaned_df[column].fillna(median_val, inplace=True)
            else:
                # For object columns, use numeric-only median
                numeric_stats = calculate_statistics_safe(cleaned_df[column])
                if numeric_stats['median'] is not None:
                    median_val = numeric_stats['median']
                    cleaned_df[column] = cleaned_df[column].fillna(median_val)
        elif missing_action == 'fill_mode':
            if cleaned_df[column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                mode_vals = cleaned_df[column].mode()
                if not mode_vals.empty:
                    mode_val = mode_vals.iloc[0]
                    cleaned_df[column].fillna(mode_val, inplace=True)
            else:
                # For object columns, try numeric mode first, then categorical mode
                numeric_stats = calculate_statistics_safe(cleaned_df[column])
                if numeric_stats['mode'] is not None:
                    mode_val = numeric_stats['mode']
                    cleaned_df[column] = cleaned_df[column].fillna(mode_val)
                else:
                    # Use categorical mode
                    mode_vals = cleaned_df[column].mode()
                    if not mode_vals.empty:
                        mode_val = mode_vals.iloc[0]
                        cleaned_df[column].fillna(mode_val, inplace=True)
        elif missing_action == 'fill_zero':
            cleaned_df[column].fillna(0, inplace=True)
        elif missing_action == 'fill_unknown':
            cleaned_df[column].fillna('Unknown', inplace=True)
        elif missing_action == 'fill_empty':
            cleaned_df[column].fillna('', inplace=True)
        elif missing_action == 'fill_false':
            cleaned_df[column].fillna(False, inplace=True)
        elif missing_action == 'fill_true':
            cleaned_df[column].fillna(True, inplace=True)
        elif missing_action == 'fill_custom':
            custom_value = options.get('custom_value', '')
            # Try to convert custom value to appropriate type
            if cleaned_df[column].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                try:
                    custom_value = float(custom_value) if '.' in str(custom_value) else int(custom_value)
                except:
                    custom_value = 0
            elif cleaned_df[column].dtype == 'bool':
                custom_value = str(custom_value).lower() in ['true', '1', 'yes', 'on']
            cleaned_df[column].fillna(custom_value, inplace=True)
    
    return cleaned_df

def auto_clean_data(df):
    """Automatically clean data with enhanced type detection and conversion"""
    cleaned_df = df.copy()
    cleaning_log = {}
    
    for column in df.columns:
        col_data = df[column]
        log_entry = {
            'original_dtype': str(col_data.dtype),
            'null_count': int(col_data.isnull().sum()),
            'action_taken': 'none',
            'method_used': 'none',
            'type_converted': False,
            'final_dtype': str(col_data.dtype)
        }
        
        # Enhanced type detection and conversion for object columns
        if col_data.dtype == 'object':
            can_convert, suggested_type, conversion_info = detect_numeric_conversion_enhanced(col_data)
            
            if can_convert and conversion_info.get('conversion_rate', 0) >= 0.85:  # 85% threshold for auto conversion
                converted_series, error = convert_column_type_enhanced(col_data, suggested_type)
                if error is None:
                    cleaned_df[column] = converted_series
                    log_entry['type_converted'] = True
                    log_entry['final_dtype'] = suggested_type
                    format_type = conversion_info.get('most_common_format', 'unknown')
                    log_entry['method_used'] = f'Auto-converted to {suggested_type} (detected {format_type} format)'
        
        # Handle missing values based on final data type
        final_col = cleaned_df[column]
        if final_col.isnull().sum() > 0:
            if final_col.dtype in ['int64', 'float64', 'Int64', 'Float64']:
                # Use median for numeric data
                median_val = final_col.median()
                cleaned_df[column].fillna(median_val, inplace=True)
                log_entry['action_taken'] = 'filled_median'
                log_entry['method_used'] += f' + filled nulls with median ({median_val:.2f})'
            else:
                # For object columns, try numeric statistics first
                numeric_stats = calculate_statistics_safe(final_col)
                if numeric_stats['count'] > 0 and numeric_stats['error'] is None:
                    # Use median of numeric values
                    median_val = numeric_stats['median']
                    cleaned_df[column].fillna(median_val, inplace=True)
                    log_entry['action_taken'] = 'filled_numeric_median'
                    log_entry['method_used'] += f' + filled nulls with numeric median ({median_val:.2f})'
                else:
                    # Use mode for categorical data
                    mode_vals = final_col.mode()
                    if not mode_vals.empty:
                        mode_val = mode_vals.iloc[0]
                        cleaned_df[column].fillna(mode_val, inplace=True)
                        log_entry['action_taken'] = 'filled_mode'
                        log_entry['method_used'] += f' + filled nulls with mode ({mode_val})'
        
        cleaning_log[column] = log_entry
    
    return cleaned_df, cleaning_log

# Legacy function names for backward compatibility
def detect_numeric_conversion(series, sample_size=10):
    """Legacy function - redirects to enhanced version"""
    return detect_numeric_conversion_enhanced(series)

def convert_column_type(series, target_type, clean_non_numeric=True):
    """Legacy function - redirects to enhanced version"""
    return convert_column_type_enhanced(series, target_type)