import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import os, json
import plotly.express as px
import plotly.graph_objects as go
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


# following is for classifier model
# def load_model():
#     """
#     Load the trained model from disk.
#     This should be called during runtime.
#     """
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train and save the model first.")
    
#     model = joblib.load(MODEL_PATH)
#     return model

# Load the trained model and tokenizer
model_path = "./model-training/trained_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Define label mapping
label_to_id = {
    "time_series": 0,
    "bar_chart": 1,
    "pie_chart": 2,
    "histogram": 3,
    "scatter_plot": 4,
    "none": 5
}
id_to_label = {v: k for k, v in label_to_id.items()}

def predict_visualization_type(question: str) -> str:
    """Predict the visualization type from the question using the fine-tuned DistilBERT model."""
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return id_to_label[predicted_class_id]

def suggest_visualizations(df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
    """
    Suggest visualizations based on model prediction, dataset structure, and question context.
    """
    structure = analyze_data_structure(df)
    suggestions = []

    # Predict the visualization type using the trained classifier model
    #visualization_type = model.predict([question])[0]
    
    # Predict the visualization type using DistilBERT
    visualization_type = predict_visualization_type(question)
    
    print(f'structure: {structure}')
    print(f'visualization_type: {visualization_type}')

    # Mapping model prediction to visualization
    if visualization_type == 'bar_chart' and structure['categorical_columns'] and structure['numeric_columns']:
        suggestions.append({
            'type': 'bar_chart',
            'possible_x': structure['categorical_columns'],
            'possible_y': structure['numeric_columns']
        })
    
    elif visualization_type == 'pie_chart' and structure['categorical_columns']:
        suggestions.append({
            'type': 'pie_chart',
            'columns': structure['categorical_columns']
        })
    
    elif visualization_type == 'histogram' and structure['numeric_columns']:
        suggestions.append({
            'type': 'histogram',
            'columns': structure['numeric_columns']
        })

    elif visualization_type == 'scatter_plot' and len(structure['numeric_columns']) >= 2:
        suggestions.append({
            'type': 'scatter_plot',
            'possible_x': structure['numeric_columns'],
            'possible_y': structure['numeric_columns']
        })

    # Handle comparison questions smartly
    comparison_keywords = ['compare', 'versus', 'vs', 'against', 'relationship']
    if any(keyword in question.lower() for keyword in comparison_keywords):
        if visualization_type == 'bar_chart' and len(structure['categorical_columns']) >= 2:
            # Prefer grouped bar charts for comparisons
            suggestions.append({
                'type': 'grouped_bar_chart',
                'possible_x': structure['categorical_columns'],
                'possible_y': structure['numeric_columns']
            })
        elif len(structure['numeric_columns']) >= 2:
            # If numeric data exists, allow scatter plots and box plots
            suggestions.append({
                'type': 'scatter_plot',
                'possible_x': structure['numeric_columns'],
                'possible_y': structure['numeric_columns']
            })
            suggestions.append({
                'type': 'box_plot',
                'possible_x': structure['numeric_columns'],
                'possible_y': structure['numeric_columns']
            })

    # Limit to top 2 suggestions for clarity
    print(f'final suggestions: {suggestions}')
    return suggestions[:2]


def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, str]) -> go.Figure:
    """
    Create a visualization based on the type and selected columns.
    """
    try:
        if viz_type == 'time_series':
            fig = px.line(df, x=columns['x'], y=columns['y'],
                         title=f'{columns["y"]} Over Time')
            
        elif viz_type == 'bar_chart':
            fig = px.bar(df, x=columns['x'], y=columns['y'],
                        title=f'{columns["y"]} by {columns["x"]}')
            
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
        
        elif viz_type == 'pie_chart':
            fig = px.pie(df, names=columns['x'], title=f'Proportion of {columns["x"]}')
            
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

