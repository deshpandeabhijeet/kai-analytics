import streamlit as st
import pandas as pd

# def select_columns(suggestion, structure, st_session_state, suggestion_index):
#     """
#     Helper function to select columns based on the visualization type.
#     """
#     cols = {}
#     col_x_key = f"{suggestion['type']}_x_{suggestion_index}"  # Append suggestion index to make key unique
#     col_y_key = f"{suggestion['type']}_y_{suggestion_index}"  # Append suggestion index to make key unique

#     # Ensure default values if keys are missing
#     st_session_state.setdefault(col_x_key, None)
#     st_session_state.setdefault(col_y_key, None)

#     if suggestion['type'] == 'time_series':
#         if suggestion.get('x_axis'):
#             st_session_state[col_x_key] = st.selectbox(
#                 "Select time column",
#                 suggestion['x_axis'],
#                 key=col_x_key  # Unique key
#             )
#         if suggestion.get('columns'):
#             st_session_state[col_y_key] = st.selectbox(
#                 "Select value column",
#                 suggestion['columns'],
#                 key=col_y_key  # Unique key
#             )
#         cols['x'] = st_session_state[col_x_key]
#         cols['y'] = st_session_state[col_y_key]

#     elif suggestion['type'] in ['bar_chart', 'scatter_plot']:
#         if suggestion.get('possible_x') and suggestion.get('possible_y'):
#             st_session_state[col_x_key] = st.selectbox(
#                 "Select X-axis",
#                 suggestion['possible_x'],
#                 key=col_x_key  # Unique key
#             )
#             st_session_state[col_y_key] = st.selectbox(
#                 "Select Y-axis",
#                 suggestion['possible_y'],
#                 key=col_y_key  # Unique key
#             )
#         cols['x'] = st_session_state[col_x_key]
#         cols['y'] = st_session_state[col_y_key]

#     elif suggestion['type'] in ['histogram', 'box_plot', 'pie_chart']:
#         # Check if 'columns' key exists in the suggestion
#         if 'columns' not in suggestion or not suggestion['columns']:
#             st.warning(f"No columns available for {suggestion['type']} visualization.")
#             return cols

#         col_key = f"{suggestion['type']}_col_{suggestion['columns'][0]}_{suggestion_index}"  # Append suggestion index
#         if col_key not in st_session_state:
#             st_session_state[col_key] = suggestion['columns'][0]

#         selected_column = st.selectbox(
#             "Select column",
#             suggestion['columns'],
#             index=suggestion['columns'].index(st_session_state[col_key]) if col_key in st_session_state and st_session_state[col_key] in suggestion['columns'] else 0,
#             key=col_key  # Unique key
#         )

#         if suggestion['type'] == 'box_plot':
#             cols['y'] = selected_column
#             group_col_key = f"{suggestion['type']}_group_col_{suggestion['columns'][0]}_{suggestion_index}"  # Append suggestion index
#             group_col = st.selectbox(
#                 "Group by (optional)",
#                 ['None'] + structure['categorical_columns'],
#                 key=group_col_key  # Unique key
#             )
#             cols['group'] = group_col if group_col != 'None' else None
#         else:
#             cols['x'] = selected_column

#     return cols


def select_columns(suggestion, structure, st_session_state, suggestion_index, df):
    """Helper function to select columns based on visualization type."""
    cols = {}
    viz_type = suggestion['type']  # Simplify variable name

    # Unique keys based on viz type and index
    col_x_key = f"{viz_type}_x_{suggestion_index}"
    col_y_key = f"{viz_type}_y_{suggestion_index}"
    col_color_key = f"{viz_type}_color_{suggestion_index}" # New color key

    # Set defaults in session state (important to handle Streamlit reruns)
    for key in [col_x_key, col_y_key, col_color_key]:
        st_session_state.setdefault(key, None)


    if viz_type == "time_series":
        cols['x'] = st.selectbox("Select time column", structure['datetime_columns'], key=col_x_key)
        cols['y'] = st.selectbox("Select value column", structure['numeric_columns'], key=col_y_key)

    elif viz_type in ['bar_chart', 'scatter_plot']:
        cols['x'] = st.selectbox("Select X-axis", df.columns, key=col_x_key) # Allow any column
        cols['y'] = st.selectbox("Select Y-axis", df.columns, key=col_y_key) # Allow any column

        #Optional color grouping for bar/scatter plots.
        cols['color'] = st.selectbox("Group/Color by (Optional)", ["None"] + df.columns.tolist(), key=col_color_key)
        if cols['color'] == "None":
            cols['color'] = None  # Remove color if None is selected


    elif viz_type in ['histogram', 'box_plot', 'pie_chart', "pie chart"]:
        print(f'df.columns: {df.columns}')
        print(f'viz_type: {viz_type}')
        print(f'col_x_key: {col_x_key}')
        cols['x'] = st.selectbox("Select column", df.columns, key=col_x_key) # Allow any column

        # Optional group for Box Plot
        if viz_type == "box_plot":
            cols['group'] = st.selectbox("Group by (optional)", ["None"] + structure['categorical_columns'], key=f"{viz_type}_group_{suggestion_index}")
            if cols['group'] == "None":
              cols['group'] = None

    return cols