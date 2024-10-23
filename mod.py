import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

class AccidentSeverityModel:
    def __init__(self, model_path):
        # Load the pre-trained model from the specified path
        with open(model_path, 'rb') as model_file:
            self.model_pipeline = pickle.load(model_file)

    def predict(self, input_data):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all necessary columns are included
        self._add_missing_columns(input_df)

        # Make a prediction and get probabilities
        prediction = self.model_pipeline.predict(input_df)
        probabilities = self.model_pipeline.predict_proba(input_df)
        
        return prediction, probabilities

    def _add_missing_columns(self, input_df):
        # Check for missing columns and add them if necessary
        for col in self.model_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0

