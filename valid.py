import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_classify_eval(file_path, threshold=0.6):
    """Load maplearn_eval.csv and classify based on threshold"""
    df_eval = pd.read_csv(file_path)
    # Filter out rows where eval_score is -1
    df_eval = df_eval[df_eval['eval_score'] != -1]
    # Convert eval_score to binary classification (0 or 1)
    df_eval['predicted'] = (df_eval['eval_score'] >= threshold).astype(int)
    return df_eval

def load_and_classify_distance(file_path):
    """Load label result file and classify based on distance"""
    df_distance = pd.read_csv(file_path)
    # Convert distance to binary classification (0 or 1)
    df_distance['actual'] = (df_distance['distance'] < 1).astype(int)

    # print(df_distance[['distance', 'actual']].head(100))

    return df_distance

def evaluate_classification(y_true, y_pred):
    """Calculate and return classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(y_true.sum(),y_pred.sum())
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

def main():
    # Load and classify evaluation data
    eval_df = load_and_classify_eval('maplearn_eval.csv', threshold=0.5)
    
    # Load and classify distance data
    distance_df = load_and_classify_distance('label_result_W30_ml_l35_556168550_20250309_123902.csv')
    
    # Merge the dataframes on ID column (assuming 'id' and 'feature_id' are the column names)
    merged_df = pd.merge(eval_df, distance_df, left_on='mpl_id', right_on='feature_id', how='inner')
    
    # Calculate metrics
    metrics = evaluate_classification(merged_df['actual'], merged_df['predicted'])
    
    # Print results
    print("\nClassification Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()