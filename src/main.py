import argparse
from train_utils.split_data import preprocess_and_split, train_cnn, preprocess_test_data
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

def train(file_path):
    (X1_train, X2_train, y_train), (X1_test, X2_test, y_test), tokenizer, max_len = preprocess_and_split(file_path)

    model = train_cnn(X1_train, X2_train, y_train, X1_test, X2_test, y_test)

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("max_len.pkl", "wb") as f:
        pickle.dump(max_len, f)

    model.save("trained_model.h5")
    print("Model, tokenizer, and max_len have been saved.")

def test(test_file_path, model_path, tokenizer_path, max_len_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(max_len_path, 'rb') as f:
        max_len = pickle.load(f)

    test_df, X1_test_data, X2_test_data = preprocess_test_data(test_file_path, tokenizer, max_len)

    model = load_model(model_path)
    print("Model loaded.")

    predictions = model.predict([X1_test_data, X2_test_data])

    test_df['PREDICTION'] = predictions
    test_df['PREDICTED_LABEL'] = (predictions > 0.5).astype(int)

    result_df = test_df[['ID', 'PREDICTED_LABEL']].copy()

    result_df.rename(columns={'PREDICTED_LABEL': 'LABEL'}, inplace=True)

    result_df.to_csv('data/test_predictions.csv', index=False)
    print("Predictions saved to data/test_predictions.csv")

def analyze(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    merged_df = pd.merge(df1[['ID', 'LABEL']], df2[['ID', 'LABEL']], on='ID', suffixes=('_ACTUAL', '_PREDICTED'))

    correct_predictions = (merged_df['LABEL_ACTUAL'] == merged_df['LABEL_PREDICTED']).sum()
    total_rows = len(merged_df)
    accuracy_percentage = (correct_predictions / total_rows) * 100

    print(f"Accuracy: {accuracy_percentage:.2f}%")

    merged_df.to_csv(output_file, index=False)

    print(f"Analysis complete. The result has been saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Train or test the CNN model.")
    parser.add_argument('mode', choices=['train', 'test', 'analyze'], help="Mode: 'train' to train the model, 'test' to run predictions, 'analyze' to analyze errors.")

    parser.add_argument('--file_path', required=True, help="Path to the dataset file (train.csv or test.csv).")
    parser.add_argument('--model_path', default='trained_model.h5', help="Path to the saved model (for testing).")
    parser.add_argument('--tokenizer_path', default='tokenizer.pkl', help="Path to the saved tokenizer (for testing).")
    parser.add_argument('--max_len_path', default='max_len.pkl', help="Path to the saved max_len (for testing).")
    parser.add_argument('--file_path2', help="Path to the second CSV file (with predicted labels).")
    parser.add_argument('--output_file', help="Path to save the analysis result (merged CSV).")

    args = parser.parse_args()

    if args.mode == 'analyze':
        if not args.file_path2 or not args.output_file:
            print("Error: Both --file_path2 and --output_file must be provided for 'analyze' mode.")
            return
        analyze(args.file_path, args.file_path2, args.output_file)
    elif args.mode == 'train':
        train(args.file_path)
    elif args.mode == 'test':
        test(args.file_path, args.model_path, args.tokenizer_path, args.max_len_path)

if __name__ == "__main__":
    main()
