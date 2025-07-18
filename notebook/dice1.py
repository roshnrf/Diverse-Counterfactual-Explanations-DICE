# Join features and target for DiCE
df_encoded = pd.concat([X_encoded, y], axis=1)

# Define DiCE data
data_dice = dice_ml.Data(
    dataframe=df_encoded,
    continuous_features=continuous_features,
    outcome_name='Diagnosis'
)

# Create DiCE model
model_dice = dice_ml.Model(model=model, backend='sklearn')
exp = Dice(data_dice, model_dice)

# Select a query instance
query_instance = X_test.iloc[[0]]  # You can change the index to choose different test instances
original_pred = model.predict(query_instance)[0]
print("Original prediction:", original_pred)

# Generate counterfactual explanations
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")

# Extract counterfactuals
cf_df = dice_exp.cf_examples_list[0].final_cfs_df
original = query_instance.reset_index(drop=True)

# Combine original and counterfactual instances
cf_combined = pd.concat([original, cf_df], ignore_index=True)
cf_combined.index = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]

# Create difference mask to highlight changes
diff_mask = cf_combined != cf_combined.iloc[0]

# Apply styling
styled = cf_combined.style.apply(
    lambda col: ['background-color: yellow' if v else '' for v in diff_mask[col.name]],
    axis=0
)

# Display the styled DataFrame
styled

# Define a function to describe changes in counterfactuals
def describe_changes(original_df, cf_df):
    explanations = []

    # Get columns that exist in both DataFrames
    common_cols = [col for col in cf_df.columns if col in original_df.columns]

    for idx, cf in cf_df.iterrows():
        changes = []
        for col in common_cols:
            orig_val = original_df[col].values[0]
            cf_val = cf[col]

            # Check if values are different and handle NaN values
            if pd.isna(orig_val) and pd.isna(cf_val):
                continue  # Both are NaN, no change
            elif pd.isna(orig_val) and not pd.isna(cf_val):
                changes.append(f"Change '{col}' from NaN to {cf_val}")
            elif not pd.isna(orig_val) and pd.isna(cf_val):
                changes.append(f"Change '{col}' from {orig_val} to NaN")
            elif orig_val != cf_val:
                changes.append(f"Change '{col}' from {orig_val} to {cf_val}")

        explanations.append(f"Counterfactual {idx+1}:\n  " + "\n  ".join(changes))
    return explanations

# Get descriptions of changes
explanations = describe_changes(original, cf_df)
for explanation in explanations:
    print(explanation)
    print()

# Additional analysis: Identify the most important features for counterfactual generation
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

    # Sort features by frequency of changes
    sorted_features = sorted(changes_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_features

# Get most important features
important_features = identify_important_features(original, cf_df)
print("Most frequently changed features in counterfactuals:")
for feature, count in important_features:
    print(f"{feature}: {count} changes")

# Visualization: Compare original vs counterfactual values for key numeric features
def plot_numeric_comparisons(original_df, cf_df, top_n=5):
    # Get numeric columns that changed in at least one counterfactual
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

    # If there are too many, limit to top_n
    if len(changed_numeric_cols) > top_n:
        changed_numeric_cols = changed_numeric_cols[:top_n]

    if changed_numeric_cols:
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(changed_numeric_cols):
            plt.subplot(1, len(changed_numeric_cols), i+1)

            # Collect all values
            values = [original_df[col].values[0]] + list(cf_df[col].values)
            labels = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]

            plt.bar(labels, values)
            plt.title(col)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

