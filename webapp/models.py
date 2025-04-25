from django.db import models
import pickle
import pandas as pd

# Load trained models
try:
    xgb = pickle.load(open(r'C:\Users\kiran\OneDrive\Desktop\CYBER_THREAT\CYBER_THREAT\FRONTEND\xgb.pkl', 'rb'))
    lgbm = pickle.load(open(r'C:\Users\kiran\OneDrive\Desktop\CYBER_THREAT\CYBER_THREAT\FRONTEND\lgbm.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")

# Load test data
data = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\CYBER_THREAT\CYBER_THREAT\FRONTEND\Test.csv')

def predict(algo, row):
    """
    Predict the class label for a given row in the test data using the selected algorithm.
    Parameters:
        algo (str): 'xgb' or 'lgbm'
        row (int): Index of the row in the test DataFrame
    Returns:
        Prediction result or None in case of error
    """
    try:
        # Extract the row as a single sample
        test_data = data.iloc[row].values.reshape(1, -1)
        
        if algo == 'xgb':
            y_pred = xgb.predict(test_data)
        elif algo == 'lgbm':
            y_pred = lgbm.predict(test_data)
        else:
            raise ValueError("Invalid algorithm choice. Choose 'xgb' or 'lgbm'.")

        return y_pred[0]  # Return scalar value instead of array
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
