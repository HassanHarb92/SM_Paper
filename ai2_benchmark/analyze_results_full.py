import pandas as pd

# Load the results file
df = pd.read_csv('full_evaluation_results_cleaned.csv')

# Extract the category (train/dev/test) from the source_file column
df['category'] = df['source_file'].apply(lambda x: x.split('-')[2].replace('.csv', '').lower())

# Check if the prediction was correct
df['correct'] = df['predicted_answer'] == df['ground_truth']

# Compute accuracy for each category
accuracy_by_category = df.groupby('category')['correct'].mean().reset_index()
accuracy_by_category['accuracy_percent'] = accuracy_by_category['correct'] * 100

# Compute overall accuracy and append as a row
overall_accuracy = df['correct'].mean() * 100
overall_row = pd.DataFrame([{'category': 'overall', 'correct': None, 'accuracy_percent': overall_accuracy}])
accuracy_by_category = pd.concat([accuracy_by_category, overall_row], ignore_index=True)

# Print results
print("\nARC Challenge Accuracy Summary:")
print(accuracy_by_category[['category', 'accuracy_percent']].to_string(index=False, float_format="%.2f"))



