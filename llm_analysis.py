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
    
    context = f"""
    Analyze this dataset based on the question: {question}
    
    Dataset Information:
    - Total rows: {structure['total_rows']}
    - Total columns: {structure['total_columns']}
    - Numeric columns: {', '.join(structure['numeric_columns'])}
    - Categorical columns: {', '.join(structure['categorical_columns'])}
    
    Sample data:
    {df.head().to_string()}
    
    Basic statistics:
    {df[structure['numeric_columns']].describe().to_string() if structure['numeric_columns'] else 'No numeric columns'}
    
    Please provide insights that are:
    1. Specific to the question asked
    2. Based on actual data patterns
    3. Include relevant numbers and statistics
    4. Suggest potential business implications if applicable
    """
    
    response = llm.invoke([HumanMessage(content=context)])
    return response.content