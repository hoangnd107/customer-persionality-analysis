import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Data loaded successfully. Shape:", self.df.shape)

    def validate_data(self):
        print("\n--- Initial Data Information ---")
        print(self.df.info())

    def handle_missing_values(self):
        if self.df.isnull().values.any():
            print("\nWarning: Dataset contains missing values. Handling missing data...")
            print("Missing values count per column:")
            print(self.df.isnull().sum())
            for column in self.df.columns:
                if self.df[column].dtype == 'object':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
            print("Missing values handled. New shape:", self.df.shape)
        else:
            print("\nNo missing values detected.")

    def drop_unnecessary_columns(self):
        if 'CustomerID' in self.df.columns:
            self.df = self.df.drop('CustomerID', axis=1)
            print("\nDropped 'CustomerID' column.")

    def encode_categorical_columns(self):
        le = LabelEncoder()
        self.df['Gender'] = le.fit_transform(self.df['Gender'])
        print("\nConverted 'Gender' to numerical format (Label Encoding).")

    def handle_outliers(self):
        numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        print("\n--- Handling Outliers ---")
        z_scores = np.abs(zscore(self.df[numerical_features]))
        threshold = 3
        outliers = (z_scores > threshold).any(axis=1)
        print(f"Outliers detected: {outliers.sum()} rows.")
        self.df = self.df[~outliers]
        print("Outliers removed. New shape:", self.df.shape)

    def scale_numerical_features(self):
        numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        scaler = StandardScaler()
        self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])
        print("\nScaled numerical features using StandardScaler.")

    def check_correlations(self):
        print("\n--- Correlation Matrix ---")
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def save_preprocessed_data(self, output_path):
        self.df.to_csv(output_path, index=False)

    def print_summary(self):
        print("\n--- First 5 Rows of Preprocessed Data ---")
        print(self.df.head())
        print("\n--- Summary of Preprocessed Data ---")
        print("Shape of dataset:", self.df.shape)
        print("Summary statistics:")
        print(self.df.describe())

preprocessor = DataPreprocessor(r"C:\Users\nduyh\py\customer-persionality-analysis\data.csv")
preprocessor.load_data()
preprocessor.validate_data()
preprocessor.handle_missing_values()
preprocessor.drop_unnecessary_columns()
preprocessor.encode_categorical_columns()
preprocessor.handle_outliers()
preprocessor.scale_numerical_features()
preprocessor.check_correlations()
preprocessor.save_preprocessed_data("predata.csv")
preprocessor.print_summary()