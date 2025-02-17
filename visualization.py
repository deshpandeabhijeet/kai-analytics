import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import streamlit as st
import numpy as np


def display_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Display data preview and basic dataset information.
    """
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Information")
    structure = analyze_data_structure(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Total Records: {structure['total_rows']:,}")
        st.write(f"Total Features: {structure['total_columns']}")
    with col2:
        st.write(f"Numeric Columns: {len(structure['numeric_columns'])}")
        st.write(f"Categorical Columns: {len(structure['categorical_columns'])}")
    return structure

def handle_visualizations(df: pd.DataFrame, viz_suggestions: List[Dict[str, Any]], structure: Dict[str, Any]):
    """Handles visualization rendering. Uses JSON data now."""
    if not viz_suggestions:
        st.write("No visualization suggestions.")
        return

    for i, viz in enumerate(viz_suggestions):
        with st.expander(f"Create {viz['type'].replace('_', ' ').title()}"):
            cols = {}  # Initialize cols for each visualization

            try:
                if viz["type"].lower() == "time_series":
                    cols['x'] = st.selectbox("Select time column", structure['datetime_columns'], key=f"time_series_x_{i}")
                    if cols['x'] is not None:
                        try:
                            df[cols['x']] = pd.to_datetime(df[cols['x']])
                        except (ValueError, TypeError):
                            st.error(f"The selected column '{cols['x']}' cannot be interpreted as a datetime. Please select a valid datetime column.")
                            continue
                    else:
                        st.warning("Please select a time column.")
                        continue
                    cols['y'] = st.selectbox("Select value column", structure['numeric_columns'], key=f"time_series_y_{i}")
                elif viz["type"].lower() in ['pie_chart', 'pie chart']:
                    handle_pie_chart(df, viz, i)
                elif "x" in viz and "y" in viz:
                    cols = {"x": viz["x"], "y": viz["y"]}
                    if "color" in viz:
                        cols["color"] = viz["color"]
                else:
                    st.warning(f"Visualization {viz.get('type')} missing required columns ('x' and 'y').")
                    continue

                if cols:
                    try:
                        missing_cols = [col for col in cols.values() if col not in df.columns and col.lower() != 'count']
                        if missing_cols:
                            st.warning(f"Skipping visualization: Columns {', '.join(missing_cols)} not found in the DataFrame.")
                            continue
                        if viz['type'] in ['bar_chart', 'bar chart'] and 'y' in cols and cols['y'].lower() == 'count' and 'count' not in df.columns:
                            fig = px.bar(df, x=cols['x'], color=cols.get('color'), title=f'Distribution of {cols["x"]}')
                        else:
                            fig, error = create_visualization(df, viz["type"], cols)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
                        elif error:
                            st.error(f"Error creating visualization: {error}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during visualization creation: {e}")
            except Exception as e:
                st.error(f"An error occurred during column selection: {e}")

def handle_pie_chart(df: pd.DataFrame, viz: Dict[str, Any], i: int):
    """Handles pie chart visualization."""
    if "data" in viz and "labels" in viz["data"] and "values" in viz["data"]:
        labels = viz["data"]["labels"]
        values = viz["data"]["values"]
        if labels is None or values is None or not all(isinstance(label, str) for label in labels):
            raise TypeError("Labels must be a list of strings")
        df_for_pie = pd.DataFrame({"labels": labels, "values": values})
        fig = px.pie(df_for_pie, names='labels', values='values', title=viz.get("options", {}).get("title"))
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
    else:
        cols = {}
        if "x" in viz and viz["x"] in df.columns:
            cols['x'] = viz['x']
        elif "y" in viz and viz["y"] in df.columns:
            cols['x'] = viz['y']
        else:
            st.warning(f"Skipping pie chart: No valid 'x' or 'y' column provided by LLM or found in the DataFrame.")
            return
        if cols:
            try:
                unique_x = df[cols['x']].dropna().unique()
                if not all(isinstance(val, str) for val in unique_x):
                    unique_x = np.array(unique_x).astype(str)
                try:
                    unique_x_sorted = sorted(unique_x)
                except TypeError:
                    unique_x_sorted = unique_x
                category_orders = {cols['x']: unique_x_sorted}
                fig = px.pie(df, names=cols['x'], title=f'Proportion of {cols["x"]}', category_orders=category_orders)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{i}")
            except (TypeError, ValueError, Exception) as e:
                st.error(f"Error creating pie chart for column '{cols['x']}': {e}")

def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization based on the type and selected columns.
    """
    try:
        if viz_type == 'time_series':
            fig = px.line(df, x=columns['x'], y=columns['y'], title=f'{columns["y"]} Over Time')
        elif viz_type in ['bar_chart','bar chart']:
            if pd.api.types.is_numeric_dtype(df[columns['x']]):
                df = df.copy()
                df[columns['x']] = df[columns['x']].astype(str)
            category_orders = {columns['x']: sorted(df[columns['x']].unique())}
            fig = px.bar(df, x=columns['x'], y=columns['y'], color=columns.get('color'),
                         title=f'{columns.get("y", "")} by {columns["x"]}', category_orders=category_orders)
        elif viz_type == 'grouped_bar_chart':
            grouped_df = df.groupby([columns['x'], columns['y']]).size().reset_index(name='count')
            fig = px.bar(grouped_df, x=columns['x'], y='count', color=columns['y'],
                         title=f'{columns["x"]} vs {columns["y"]}', barmode='group')
        elif viz_type == 'heatmap':
            heatmap_data = df.groupby([columns['x'], columns['y']]).size().unstack()
            fig = px.imshow(heatmap_data, labels=dict(x=columns['x'], y=columns['y'], color='Count'),
                            title=f'{columns["x"]} vs {columns["y"]}')
        elif viz_type == 'histogram':
            fig = px.histogram(df, x=columns['x'], title=f'Distribution of {columns["x"]}')
        elif viz_type == 'box_plot':
            if 'group' in columns:
                fig = px.box(df, x=columns['group'], y=columns['y'],
                             title=f'Distribution of {columns["y"]} by {columns["group"]}')
            else:
                fig = px.box(df, y=columns['y'], title=f'Distribution of {columns["y"]}')
        elif viz_type.lower() in ['scatter_plot', 'scatter plot']:
            fig = px.scatter(df, x=columns['x'], y=columns['y'],
                             title=f'Relationship between {columns["x"]} and {columns["y"]}')
        elif viz_type.lower() in ['pie_chart','pie chart']:
            if "x" in columns and columns["x"] in df.columns:
                if pd.api.types.is_numeric_dtype(df[columns['x']]):
                    df = df.copy()
                    df[columns['x']] = df[columns['x']].astype(str)
                unique_x = df[columns['x']].unique()
                if isinstance(unique_x, np.ndarray):
                    unique_x = list(map(str, unique_x))
                elif isinstance(unique_x, pd.Series):
                    unique_x = list(map(str, unique_x.tolist()))
                category_orders = {columns['x']: sorted(unique_x)}
                fig = px.pie(df, names=columns["x"], category_orders=category_orders,
                             title=f'Proportion of {columns["x"]}')
            elif "y" in columns and columns["y"] in df.columns:
                if pd.api.types.is_numeric_dtype(df[columns['y']]):
                    df = df.copy()
                    df[columns['y']] = df[columns['y']].astype(str)
                fig = px.pie(df, names=columns["y"], title=f'Proportion of {columns["y"]}')
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