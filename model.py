# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import pickle

# Load dataset
df = pd.read_csv(r'C:/Users/rithi/Downloads/accidents_2012_to_2014.csv (1)/accidents_2012_to_2014.csv', nrows=1000, low_memory=False)

# Step 1: Analyze the time of accidents
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
sns.countplot(x='Time', data=df)
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.show()

# Step 2: Preprocessing - Remove irrelevant columns, keeping only necessary features
columns_to_keep = ['Latitude', 'Longitude', 'Time', 'Day_of_Week', 'Weather_Conditions', 'Accident_Severity']
df = df[columns_to_keep]

# Step 3: Handle missing values in 'Time' column
df['Time'] = df['Time'].fillna(df['Time'].mean())

# Step 4: One-hot encode categorical variables (Day_of_Week, Weather_Conditions)
df = pd.get_dummies(df, columns=['Day_of_Week', 'Weather_Conditions'], drop_first=True)

# Step 5: Check unique values in the target variable
print("Unique values in target variable before adjustment:", df['Accident_Severity'].unique())

# Adjust target variable if necessary (e.g., from [1, 2, 3] to [0, 1, 2])
df['Accident_Severity'] = df['Accident_Severity'] - 1

# Step 6: Define features and target variable
X = df.drop(columns=['Accident_Severity'])
y = df['Accident_Severity']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Check the class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y_train.value_counts())

# Step 9: Apply SMOTE to the training data with a reduced n_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 10: Display the results of SMOTE
print("Before SMOTE:")
print(y_train.value_counts())
print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

# Step 11: Store the resampled distribution for plotting
resampled_distribution = y_train_resampled.value_counts()

# Step 12: Plot class distribution before and after SMOTE
plt.figure(figsize=(10, 6))

# Bar plot for before SMOTE
plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind='bar', color='blue', alpha=0.7)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Accident Severity')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)

# Bar plot for after SMOTE
plt.subplot(1, 2, 2)
resampled_distribution.plot(kind='bar', color='orange', alpha=0.7)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Accident Severity')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Step 13: Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), X.select_dtypes(include=['int64', 'float64']).columns)
    ]
)

# Step 14: Define models for individual classifiers
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Step 15: Training each model individually
for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Preprocess the data
        ('classifier', model)            # Train the model
    ])
    
    pipeline.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate each model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    print(f"{model_name} - Classification Report:")
    print(classification_report(y_test, y_pred))

# Step 16: Define meta-model
meta_model = LogisticRegression()

# Initialize the base models for stacking
base_models = [
    ('RandomForest', models['RandomForest']),
    ('GradientBoosting', models['GradientBoosting']),
    ('SVC', models['SVC']),
    ('XGBoost', models['XGBoost'])
]

# Step 17: Create the stacking classifier
stacking_clf = StackingClassifier(classifiers=[model for name, model in base_models], 
                                  meta_classifier=meta_model, 
                                  use_probas=True,  # Use probabilities from the base models
                                  average_probas=False)

# Step 18: Create the pipeline for stacking (with preprocessing)
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocess the data
    ('stacking', stacking_clf)       # Apply stacking
])

# Step 19: Train stacking model
stacking_pipeline.fit(X_train_resampled, y_train_resampled)

# Step 20: Evaluate stacking model on test data
y_pred_stack = stacking_pipeline.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)

print(f"Stacking Model - Accuracy: {accuracy_stack:.4f}")
print(f"Stacking Model - Classification Report:")
print(classification_report(y_test, y_pred_stack))

# Step 21: Save the stacking model to a file
with open('stacking_model.pkl', 'wb') as model_file:
    pickle.dump(stacking_pipeline, model_file)

print("Stacking model saved as 'stacking_model.pkl'")

# Step 22: Load the saved model from the .pkl file
with open('stacking_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Step 23: Function to predict accident severity using the loaded model
def predict_accident_severity(input_data):
    # Convert input_data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all one-hot encoded columns are present, add missing ones
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default value 0

    # Make prediction
    prediction = loaded_model.predict(input_df)

    # Get probability
    probability = loaded_model.predict_proba(input_df)

    return {
        'severity': prediction[0],
        'probability': probability[0].max()
    }
