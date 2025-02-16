import numpy as np
import pandas as pd
from typing import Dict, Any, List

import plotly.express as px
import plotly.graph_objects as go

def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization based on the type and selected columns.
    """
    print(f'columns: {columns}')
    print(f'viz_type: {viz_type}')
    try:
        if viz_type == 'time_series':
            fig = px.line(df, x=columns['x'], y=columns['y'],
                         title=f'{columns["y"]} Over Time')
            
        elif viz_type in ['bar_chart','bar chart']:
            # Correct handling for bar charts: Only convert to string if numeric
            if pd.api.types.is_numeric_dtype(df[columns['x']]):
                df = df.copy()  # Create a copy!
                df[columns['x']] = df[columns['x']].astype(str)
            category_orders = {columns['x']: sorted(df[columns['x']].unique())}
            fig = px.bar(df, x=columns['x'], y=columns['y'], color=columns.get('color'),
                    title=f'{columns.get("y", "")} by {columns["x"]}', category_orders=category_orders)
            
        elif viz_type == 'grouped_bar_chart':
            # Grouped bar chart for comparing two categorical columns
            grouped_df = df.groupby([columns['x'], columns['y']]).size().reset_index(name='count')
            fig = px.bar(grouped_df, x=columns['x'], y='count', color=columns['y'],
                        title=f'{columns["x"]} vs {columns["y"]}',
                        barmode='group')
            
        elif viz_type == 'heatmap':
            # Heatmap for comparing two categorical columns
            heatmap_data = df.groupby([columns['x'], columns['y']]).size().unstack()
            fig = px.imshow(heatmap_data, labels=dict(x=columns['x'], y=columns['y'], color='Count'),
                           title=f'{columns["x"]} vs {columns["y"]}')
            
        elif viz_type == 'histogram':
            fig = px.histogram(df, x=columns['x'],
                             title=f'Distribution of {columns["x"]}')
            
        elif viz_type == 'box_plot':
            if 'group' in columns:
                fig = px.box(df, x=columns['group'], y=columns['y'],
                           title=f'Distribution of {columns["y"]} by {columns["group"]}')
            else:
                fig = px.box(df, y=columns['y'],
                           title=f'Distribution of {columns["y"]}')
                
        elif viz_type == 'scatter_plot':
            fig = px.scatter(df, x=columns['x'], y=columns['y'],
                           title=f'Relationship between {columns["x"]} and {columns["y"]}')
        elif viz_type == 'pie_chart':  # Complete pie chart handling
            print('pie chart handling inside create_visualization')
            if "x" in columns and columns["x"] in df.columns: # Correctly use 'columns', handle numeric 'x'
                if pd.api.types.is_numeric_dtype(df[columns['x']]):
                    df = df.copy()  # Create a copy of the DataFrame
                    df[columns['x']] = df[columns['x']].astype(str)

                # Ensure correct type for category_orders
                unique_x = df[columns['x']].unique()
                if isinstance(unique_x, np.ndarray):
                    unique_x = list(map(str, unique_x)) #Make it hashable
                elif isinstance(unique_x, pd.Series):
                    unique_x = list(map(str, unique_x.tolist()))

                category_orders = {columns['x']: sorted(unique_x)}  # Now should be hashable

                fig = px.pie(df, names=columns["x"], category_orders=category_orders,
                             title=f'Proportion of {columns["x"]}')
            elif "y" in columns and columns["y"] in df.columns: # Correctly use 'columns', handle numeric 'y'
                if pd.api.types.is_numeric_dtype(df[columns['y']]):
                    df = df.copy() #Make a copy of DF.
                    df[columns['y']] = df[columns['y']].astype(str)
                fig = px.pie(df, names=columns["y"],
                             title=f'Proportion of {columns["y"]}')
            else:
                return None, "Pie chart requires either 'data' with 'labels' and 'values', or a valid 'x' or 'y' column."

            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        return fig, None
        
    except Exception as e:
        return None, str(e)

def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the structure of the dataframe to understand its contents
    """
    structure = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=['float64', 'int64']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean_columns': list(df.select_dtypes(include=['bool']).columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return structure

