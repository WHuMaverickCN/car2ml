import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml

def load_and_classify_eval(file_path, threshold=0.7,agg_count = 8,floater = -0.14):
    """Load maplearn_eval.csv and classify based on threshold"""
    df_eval = pd.read_csv(file_path)
    # Filter out rows where eval_score is -1
    df_eval = df_eval[df_eval['eval_score'] != -1]
    # Convert eval_score to binary classification (0 or 1) based on aggre_count
    threshold_2 = threshold + floater  # You can adjust this value as needed
    df_eval['predicted'] = ((df_eval['aggre_count'] < agg_count) & (df_eval['eval_score'] >= threshold) | 
                          (df_eval['aggre_count'] >= agg_count) & (df_eval['eval_score'] >= threshold_2)).astype(int)
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
    negative_recall = recall_score(y_true, y_pred, pos_label=0)  # Recall for negative class
    negative_precision = precision_score(y_true, y_pred, pos_label=0)  # Precision for negative class
    f1 = f1_score(y_true, y_pred)
    negative_f1 = f1_score(y_true, y_pred, pos_label=0)  # F1 score for negative class
    conf_matrix = confusion_matrix(y_true, y_pred)
    # print(y_true.sum(),y_pred.sum())
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'negative_recall': negative_recall,
        'negative_precision': negative_precision,
        'negative_f1': negative_f1,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

def main():
    # _label = "./data/valid/label_result_W30_ml_l35_556168550_20250309_123902.csv"
    # _pred = "./data/valid/maplearn_eval_556168550.csv"


    # Read config file
    with open('config_bounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    _label = config['valid']['label_path']
    _pred = config['valid']['pred_path']
    # _label = "label_result_W30_ml_l35_556168546_20250311_171520.csv"
    # _pred = "maplearn_eval.csv"

    # _label = "./data/valid/label_result_W30_ml_l35_556168503_20250311_170557.csv"
    # _pred = "./data/valid/maplearn_eval_556168503.csv"

    '''
    # Load and classify evaluation data
    # Define parameter ranges
    # thresholds = [round(x/10, 1) for x in range(3, 11)]  # 0.3 to 1.0
    # agg_counts = list(range(2, 20))  # 2 to 19
    # floaters = [round(x/100, 2) for x in range(-30, 31)]  # -0.3 to 0.3 with 0.01 steps

    # thresholds = [0.7,0.6,0.5]
    # agg_counts = [4,5,8,9]
    # floaters = [0.22,0.26,0.12,0.06]
    '''

    thresholds = [0.6]
    agg_counts = [8]
    floaters = [0.22]

    # Load distance data once
    distance_df = load_and_classify_distance(_label)

    # Loop through all parameter combinations
    best_negative_f1 = 0  # Initialize the best negative recall score

    for threshold in thresholds:
        for agg_count in agg_counts:
            for floater in floaters:
                print(f"\nParameters: threshold={threshold}, agg_count={agg_count}, floater={floater}")
                
                # Load and classify evaluation data with current parameters
                eval_df = load_and_classify_eval(
                    file_path=_pred,
                    threshold=threshold,
                    agg_count=agg_count,
                    floater=floater
                )
                
                # Merge the dataframes
                merged_df = pd.merge(eval_df, distance_df, left_on='mpl_id', right_on='feature_id', how='inner')
                
                # Calculate and print metrics
                metrics = evaluate_classification(merged_df['actual'], merged_df['predicted'])
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"negative_f1: {metrics['negative_f1']:.4f}")
                print(f"negative_recall:{metrics['negative_recall']:.4f}")
                print(f"F1 Score:{metrics['f1_score']:.4f}")
                print("Confusion Matrix:")
                print(metrics['confusion_matrix'])


                # if metrics['accuracy'] >= 0.80 and metrics['negative_f1'] > best_negative_f1:
                #     best_negative_f1 = metrics['negative_f1']
                #     print(f"\n!!! New best negative recall: {best_negative_f1:.4f} with accuracy {metrics['accuracy']:.4f} !!!")
                #     print(f"Parameters: threshold={threshold}, agg_count={agg_count}, floater={floater}")
                #     input("Press Enter to continue...")
                
                

if __name__ == "__main__":
    main()