import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from visualization import analyze_data_structure
import pandas as pd

def analyze_with_llm(df: pd.DataFrame, question: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Get insights from LLM based on the question and data
    """
    # Prepare data context
    structure = analyze_data_structure(df)
    sample_data = df.sample(n=min(len(df), 5))
    available_columns = ', '.join(df.columns)
    
    context = f"""
    Analyze this dataset based on the question: {question}
    
    Dataset Information:
    - Total rows: {structure['total_rows']}
    - Total columns: {structure['total_columns']}
    - Numeric columns: {', '.join(structure['numeric_columns'])}
    - Categorical columns: {', '.join(structure['categorical_columns'])}
    
    First 5 records:
    {df.head().to_string()}
    
    Sample 5 records:
    {sample_data.to_string()}
    
    Basic statistics:
    {df[structure['numeric_columns']].describe().to_string() if structure['numeric_columns'] else 'No numeric columns'}
    
    Please provide insights that are:
    1. Specific to the question asked
    2. Based on actual data patterns
    3. Include relevant numbers and statistics
    4. Suggest potential business implications if applicable.
                    
    Suggest 1 or 2 appropriate visualizations. 
    For each visualization, specify the chart type (e.g., bar chart, scatter plot, line graph) and which columns should be used for the x and y axes (and any other relevant parameters). Explain your reasoning for each suggestion.
    
    Available columns: {available_columns}
    When suggesting visualizations, use only the available columns listed above.  Do not suggest visualizations that use columns not present in the data.

    Return your analysis and visualization suggestions in the following strict JSON format:

    {{
      "insights": "Your insightful analysis here...",
       "visualizations": [
        {{
          "type": "chart_type",  # Generic chart type
          "x": "column_name",
          "y": "column_name",
          // ... other parameters as needed
        }},
        // ... more visualizations if appropriate
      ]
    }}
    
    When suggesting visualizations, consider these guidelines:
    - For distributions of a single categorical variable, pie charts or histograms are often preferred.
    - Bar charts are generally suitable for comparing values across different categories.
    - If there's a time component, suggest a time series chart.
    - Scatter plots are useful for exploring relationships between two numerical variables.
    - Choose the chart type that best represents the data and answers the question.
    
    Do not include any markdown or extra formatting. The only output should be valid JSON.  Ensure the entire response is parseable as a single JSON object.
    """
    
    response = llm.invoke([HumanMessage(content=context)])
    #print(f'llm response: {response}')
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