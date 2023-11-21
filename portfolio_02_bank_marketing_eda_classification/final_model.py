# Import libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import pandas as pd
import numpy as np

input_directory = input("Paste the path of the new_data.csv file: ").strip()

# Create dataframe
df = pd.read_csv(input_directory, delimiter=';')

# Define groups of cols
numeric_features =  df[['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']]
binary_cols = df[['default', 'housing', 'loan', 'y']]
other_categorical_cols = df[['job', 'marital', 'education', 'contact', 'poutcome']]

# Define month mapping
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# Create the preprocessing steps for the ColumnTransformer
binary_transformer = Pipeline(steps=[
    ('binary_encoder', StandardScaler())
])

other_categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Bundle preprocessing for binary and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_cols),
        ('cat', other_categorical_transformer, other_categorical_cols)
    ])

# Create and configure the final model
final_model = RandomForestClassifier(
    class_weight={0: 0.2, 1: 0.8},
    max_depth=None,
    max_features=0.5,
    min_samples_leaf=5,
    min_samples_split=15,
    n_estimators=150,
    random_state=42
)

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', final_model)]

# Split the data
X = df.drop(columns=['y']).copy()
y = df['y'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)


def make_predictions(new_data, pipeline):
    # Ensure the new data has the same column names as the original data
    expected_columns = set(pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].get_feature_names_out())
    new_data = new_data[expected_columns]

    # Make predictions on the new data
    y_pred = pipeline.predict(new_data)

    # Add the predictions to the new_data DataFrame
    new_data['predicted_y'] = y_pred

    return new_data

if __name__ == '__main':
    # Input the new data
    input_directory = input("Paste the path of the new_data.csv file: ").strip()
    new_data = pd.read_csv(input_directory, delimiter=';')

    # Load your pre-trained pipeline
    # Example: pipeline = load_your_pretrained_model()

    # Make predictions
    results = make_predictions(new_data, pipeline)

    # Output the results
    output_file = input("Enter the path to save the results CSV file: ")
    results.to_csv(output_file, index=False)
