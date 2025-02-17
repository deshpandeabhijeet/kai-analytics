import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from visualization import analyze_data_structure
import pandas as pd
import streamlit as st

def analyze_with_llm(df: pd.DataFrame, question: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Get insights from LLM based on the question and data
    """
    # Prepare data context
    structure = analyze_data_structure(df)
    sample_data = df.sample(n=min(len(df), 5))
    available_columns = ', '.join(df.columns)
    
    context = f"""
    Analyze the following dataset based on the question: {question}

    ### Dataset Information:
    - Total rows: {structure['total_rows']}
    - Total columns: {structure['total_columns']}
    - Numeric columns: {', '.join(structure['numeric_columns'])}
    - Categorical columns: {', '.join(structure['categorical_columns'])}

    ### Data Preview:
    - First 5 records:
    {df.head().to_string()}

    - Sample 5 records:
    {sample_data.to_string()}

    ### Basic Statistics:
    {df[structure['numeric_columns']].describe().to_string() if structure['numeric_columns'] else 'No numeric columns'}

    ### Instructions for Analysis:
    1. **Specificity**: Ensure insights directly address the question asked.
    2. **Data Patterns**: Base insights on actual trends, correlations, or anomalies in the data.
    3. **Numbers and Statistics**: Include relevant metrics, percentages, or statistical findings to support your insights.
    4. **Business Implications**: Suggest potential implications or actionable recommendations for the business, if applicable.

    ### Visualization Guidelines:
    - Suggest 1 or 2 visualizations that best represent the data and answer the question.
    - For each visualization:
    - Specify the chart type (e.g., bar chart, scatter plot, line graph).
    - Indicate which columns should be used for the x and y axes (and any other relevant parameters).
    - Explain why the chosen visualization is appropriate for the data and question.
    - Use only the available columns: {available_columns}. Do not suggest visualizations that require columns not present in the dataset.

    ### Output Format:
    Return your analysis and visualization suggestions in the following strict JSON format:

    {{
    "insights": [
        {{
        "specificity": "Insight directly addressing the question...",
        "data_patterns": "Description of trends, correlations, or anomalies...",
        "numbers_and_statistics": "Relevant metrics or statistics...",
        "business_implications": "Potential implications or recommendations..."
        }},
        // Add additional insights if applicable
    ],
    "visualizations": [
        {{
        "type": "chart_type",  # Generic chart type
        "x": "column_name",
        "y": "column_name",
        "reasoning": "Explanation of why this visualization is appropriate..."
        }},
        // Add additional visualizations if applicable
    ]
    }}

    ### Additional Notes:
    - Ensure the response is parseable as a single JSON object.
    - Do not include any markdown or extra formatting outside the JSON structure.
    - If no numeric columns are available, focus on categorical analysis and suggest appropriate visualizations (e.g., bar charts, pie charts).
    - If the dataset is small or lacks clear patterns, state this explicitly in the insights and suggest exploratory visualizations.

    """
    
    response = llm.invoke([HumanMessage(content=context)])
    #print(f'llm raw response: {response}')
    try:
        print(f"inside response parsing")
        cleaned_response = response.content.replace("`", "")
        cleaned_response = response.content.replace("```json", "").replace("```", "")
        cleaned_response = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned_response)
        cleaned_response = re.sub(r'("insights":\s*")([\s\S]*?)(")', lambda m: f'{m.group(1)}{m.group(2).replace("\n", " ")}{m.group(3)}', cleaned_response)

        cleaned_response = cleaned_response.replace("\"", "\\\"")
        cleaned_response = cleaned_response.replace("\\\"", "\"")
        # cleaned_response = cleaned_response.replace("**Insights:**", "")
        # cleaned_response = cleaned_response.replace("**Visualizations:**", "")
        # cleaned_response = cleaned_response.replace("**Potential business implications:**", "")

        cleaned_response = cleaned_response.strip()

        #print(f"cleaned response, before json.loads:\n{cleaned_response}")
        json_data = json.loads(cleaned_response)
        print(f"Valid JSON received (after cleaning):\n{json.dumps(json_data, indent=2)}")
        return json_data
    except json.JSONDecodeError:
      # Graceful fallback if the LLM doesn't return valid JSON
      print(f"Invalid JSON returned from LLM: {response.content}")
      print(f"JsonDecode Error: {json.JSONDecodeError}")
      return {
          "insights": response.content,  # Return the raw response as insights
          "visualizations": []  # Empty list of visualizations
      }
      

def get_insights_and_visualization_suggestions(df, question, llm):
    """Gets insights and visualizations, now handles JSON parsing."""
    if question and question != st.session_state.get("last_question", ""):
        with st.spinner("Analyzing..."):
            response = analyze_with_llm(df, question, llm)
            st.session_state["last_question"] = question
            st.session_state["last_response"] = response # Store the full, parsed JSON response
    else:
        response = st.session_state.get("last_response", {}) # Access potentially cached response

    insights = response.get("insights", "") # Access elements safely
    viz_suggestions = response.get("visualizations", [])
    return insights, viz_suggestions