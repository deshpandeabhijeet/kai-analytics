# Setup and Run the application on local. 

## Create virtual environment
``python -m venv venv``

## Install dependencies
``pip install -r requirements.txt``

## Train the distil-bert model 
### Navigate the right folder 
``cd trained_model``
### run the fine-tuning code
``python fine-tuning.py``

# Run the streamlit application
``streamlit run app.py``
