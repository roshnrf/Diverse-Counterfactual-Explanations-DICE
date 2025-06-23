# -*- coding: utf-8 -*-
"""dice1.ipynb

I have included necessery comments for your understanding :)

Original file is located at
    https://colab.research.google.com/drive/1kDzzsKX2AIH63kWnrvntnpdq0Bo77Cfm
"""

# First, let's install dice-ml and compatible pandas/numpy versions.
# dice-ml is the library for generating counterfactual explanations.
!pip install dice-ml
!pip install pandas==1.5.3 numpy==1.24.3 --force-reinstall

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import dice_ml
from dice_ml import Dice

# Load your dataset.
# This code assumes your data is in a CSV file named 'alzheimers_disease_data.csv', if using some otehr dataset you have change other factors too!!!
# The file picker dialog will open for you to upload the file.
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("alzheimers_disease_data.csv")

# Fill missing values in the dataset.
# For categorical columns, fill with the most frequent value (mode).
# For numerical columns, fill with the median.
for col in df.columns:
    if df[col].isna().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Select features for modeling.
# Exclude identifiers and columns not used for prediction.
features = [col for col in df.columns if col not in ['PatientID', 'DoctorInCharge', 'Diagnosis', 'Gender', 'Ethnicity','EducationLevel']]
X = df[features]
y = df['Diagnosis']  # Target variable (assumed to be binary: 0 or 1)

# Define which features are categorical for one-hot encoding.
categorical_features = ['FamilyHistoryAlzheimers',
                        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
                        'Hypertension', 'MemoryComplaints', 'BehavioralProblems',
                        'Confusion', 'Disorientation', 'PersonalityChanges',
                        'DifficultyCompletingTasks', 'Forgetfulness', 'Smoking']

# List of continuous (numeric) features.
continuous_features = [col for col in features if col not in categorical_features]

# Convert categorical features to one-hot encoding for model compatibility.
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split the data into training and test sets.
# test_size=0.3 means 30% of data is used for testing.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Train a logistic regression model.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the test set.
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot the confusion matrix to visualize prediction results.
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Prepare the data for DiCE.
# DiCE requires both features and the outcome variable in a single DataFrame.
df_encoded = pd.concat([X_encoded, y], axis=1)

# Define the DiCE data object.
# continuous_features: list of numeric features.
# outcome_name: the target variable.
data_dice = dice_ml.Data(
    dataframe=df_encoded,
    continuous_features=continuous_features,
    outcome_name='Diagnosis'
)

# Create a DiCE model object using the trained sklearn model.
model_dice = dice_ml.Model(model=model, backend='sklearn')
exp = Dice(data_dice, model_dice)

# Select a query instance (a single row from the test set) for which to generate counterfactuals.
# You can change the index to select a different instance.
query_instance = X_test.iloc[[0]]
original_pred = model.predict(query_instance)[0]
print("Original prediction:", original_pred)

# Generate counterfactual explanations.
# total_CFs: number of counterfactuals to generate.
# desired_class: 'opposite' means generate counterfactuals that flip the predicted class.
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")

# Extract the counterfactuals as a DataFrame.
cf_df = dice_exp.cf_examples_list[0].final_cfs_df
original = query_instance.reset_index(drop=True)

# Combine the original instance and counterfactuals for comparison.
cf_combined = pd.concat([original, cf_df], ignore_index=True)
cf_combined.index = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]

# Create a mask to highlight which features changed in counterfactuals.
diff_mask = cf_combined != cf_combined.iloc[0]

# Style the DataFrame to visually highlight changes (yellow background).
styled = cf_combined.style.apply(
    lambda col: ['background-color: yellow' if v else '' for v in diff_mask[col.name]],
    axis=0
)

# Display the styled DataFrame.
styled

# Function to describe what changed in each counterfactual.
def describe_changes(original_df, cf_df):
    explanations = []

    # Only compare columns present in both DataFrames.
    common_cols = [col for col in cf_df.columns if col in original_df.columns]
    for idx, cf in cf_df.iterrows():
        changes = []
        for col in common_cols:
            orig_val = original_df[col].values[0]
            cf_val = cf[col]

            # Handle NaN values and differences.
            if pd.isna(orig_val) and pd.isna(cf_val):
                continue
            elif pd.isna(orig_val) and not pd.isna(cf_val):
                changes.append(f"Change '{col}' from NaN to {cf_val}")
            elif not pd.isna(orig_val) and pd.isna(cf_val):
                changes.append(f"Change '{col}' from {orig_val} to NaN")
            elif orig_val != cf_val:
                changes.append(f"Change '{col}' from {orig_val} to {cf_val}")

        explanations.append(f"Counterfactual {idx+1}:\n  " + "\n  ".join(changes))
    return explanations

# Print the explanations for each counterfactual.
explanations = describe_changes(original, cf_df)
for explanation in explanations:
    print(explanation)
    print()

# Function to identify which features are most frequently changed in counterfactuals.
def identify_important_features(original_df, cf_df):
    changes_count = {}

    common_cols = [col for col in cf_df.columns if col in original_df.columns]

    for idx, cf in cf_df.iterrows():
        for col in common_cols:
            orig_val = original_df[col].values[0]
            cf_val = cf[col]

            if orig_val != cf_val:
                if col in changes_count:
                    changes_count[col] += 1
                else:
                    changes_count[col] = 1

    # Sort features by how often they change.
    sorted_features = sorted(changes_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_features

# Print the most frequently changed features.
important_features = identify_important_features(original, cf_df)
print("Most frequently changed features in counterfactuals:")
for feature, count in important_features:
    print(f"{feature}: {count} changes")

# Function to plot numeric feature changes between the original and counterfactuals.
def plot_numeric_comparisons(original_df, cf_df, top_n=5):
    # Find numeric columns that changed.
    numeric_cols = [col for col in cf_df.columns
                   if col in original_df.columns
                   and np.issubdtype(original_df[col].dtype, np.number)]

    changes = {}
    for col in numeric_cols:
        orig_val = original_df[col].values[0]
        changed = False
        for _, cf in cf_df.iterrows():
            if orig_val != cf[col]:
                changed = True
                break
        if changed:
            changes[col] = True

    changed_numeric_cols = list(changes.keys())

    # Limit to top_n columns if too many.
    if len(changed_numeric_cols) > top_n:
        changed_numeric_cols = changed_numeric_cols[:top_n]

    if changed_numeric_cols:
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(changed_numeric_cols):
            plt.subplot(1, len(changed_numeric_cols), i+1)

            # Collect values for plotting.
            values = [original_df[col].values[0]] + list(cf_df[col].values)
            labels = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]

            plt.bar(labels, values)
            plt.title(col)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

# Plot numeric comparisons for the most changed features.
plot_numeric_comparisons(original, cf_df)

# Example: Try generating counterfactuals with different proximity weights.
# proximity_weight controls how much the counterfactuals should resemble the original instance.
# Higher values make counterfactuals closer to the original.
for weight in [0.1, 0.5, 1.0]:
    print(f"\nGenerating counterfactuals with proximity_weight={weight}")
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class="opposite",
        proximity_weight=weight
    )

    # Print the explanations for each set of counterfactuals.
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    explanations = describe_changes(original, cf_df)
    for explanation in explanations:
        print(explanation)
