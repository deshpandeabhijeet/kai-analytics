import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List

from data_processing import analyze_data_structure

def suggest_visualizations(df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
    """
    Suggest appropriate visualizations based on the question and data types.
    """
    question = question.lower()
    structure = analyze_data_structure(df)
    suggestions = []
    
    # Keywords that suggest certain visualization types
    time_keywords = ['trend', 'over time', 'historical', 'growth', 'changes']
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'against']
    distribution_keywords = ['distribution', 'spread', 'range', 'histogram']
    relationship_keywords = ['correlation', 'relationship', 'between', 'affect']
    categorical_keywords = ['category', 'group', 'segment', 'proportion', 'percent']

    # Check for time-based analysis
    if any(keyword in question for keyword in time_keywords) and structure['datetime_columns']:
        suggestions.append({
            'type': 'time_series',
            'columns': structure['numeric_columns'],
            'x_axis': structure['datetime_columns'][0]
        })
    
    # Check for categorical distributions
    if any(keyword in question for keyword in categorical_keywords) and structure['categorical_columns']:
        suggestions.append({
            'type': 'pie_chart',
            'columns': structure['categorical_columns']
        })
    
    # Check for comparisons
    if any(keyword in question for keyword in comparison_keywords):
        if structure['categorical_columns'] and structure['numeric_columns']:
            suggestions.append({
                'type': 'bar_chart',
                'possible_x': structure['categorical_columns'],
                'possible_y': structure['numeric_columns']
            })
    
    # Check for distributions
    if any(keyword in question for keyword in distribution_keywords):
        if structure['numeric_columns']:
            suggestions.append({
                'type': 'histogram',
                'columns': structure['numeric_columns']
            })
            suggestions.append({
                'type': 'box_plot',
                'columns': structure['numeric_columns']
            })
    
    # Check for relationships
    if any(keyword in question for keyword in relationship_keywords):
        if len(structure['numeric_columns']) >= 2:
            suggestions.append({
                'type': 'scatter_plot',
                'possible_x': structure['numeric_columns'],
                'possible_y': structure['numeric_columns']
            })
    
    return suggestions

def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, str]) -> go.Figure:
    """
    Create a visualization based on the type and selected columns
    """
    try:
        if viz_type == 'time_series':
            fig = px.line(df, x=columns['x'], y=columns['y'],
                         title=f'{columns["y"]} Over Time')
            
        elif viz_type == 'bar_chart':
            fig = px.bar(df, x=columns['x'], y=columns['y'],
                        title=f'{columns["y"]} by {columns["x"]}')
            
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
        
        elif viz_type == 'pie_chart':
            fig = px.pie(df, names=columns['x'], title=f'Proportion of {columns["x"]}')
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        return fig, None
        
    except Exception as e:
        return None, str(e)