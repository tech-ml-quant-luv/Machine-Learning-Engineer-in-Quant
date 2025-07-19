# data_preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        # Configuration placeholders
        self.file_path = None
        self.target_column = -1
        self.impute_columns = (1, 2)
        self.impute_strategy = "mean"
        self.onehot_column = 0
        self.label_encode_target = True
        self.test_size = 0.2
        self.random_state = 1
        self.scale_columns_from = 1
        self.delimiter = ','

        # Internal objects
        self.dataset = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # Transformers
        self.imputer = None
        self.label_encoder_Y = LabelEncoder()
        self.scaler = StandardScaler()
        self.column_transformer = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        print("[INFO] Parameters updated.")
        return self

    def load_data(self):
        self.dataset = pd.read_csv(self.file_path, delimiter=self.delimiter)
        self.X = self.dataset.iloc[:, :-1].values
        self.Y = self.dataset.iloc[:, self.target_column].values
        print("[INFO] Data loaded successfully.")
        return self.X, self.Y

    def handle_missing_data(self):
        self.imputer = SimpleImputer(missing_values=float("nan"), strategy=self.impute_strategy)
        self.X[:, self.impute_columns[0]:self.impute_columns[1]+1] = self.imputer.fit_transform(
            self.X[:, self.impute_columns[0]:self.impute_columns[1]+1]
        )
        print("[INFO] Missing values handled.")
        return self.X

    def encode_features(self):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), [self.onehot_column])
            ],
            remainder='passthrough'
        )
        self.X = self.column_transformer.fit_transform(self.X)
        print("[INFO] One-hot encoding applied to features.")
        return self.X

    def encode_target(self):
        if self.label_encode_target:
            self.Y = self.label_encoder_Y.fit_transform(self.Y)
            print("[INFO] Target variable label-encoded.")
        return self.Y

    def split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=self.random_state
        )
        print("[INFO] Data split into training and test sets.")
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def scale_features(self):
        self.X_train[:, self.scale_columns_from:] = self.scaler.fit_transform(self.X_train[:, self.scale_columns_from:])
        self.X_test[:, self.scale_columns_from:] = self.scaler.transform(self.X_test[:, self.scale_columns_from:])
        print("[INFO] Features scaled from column index", self.scale_columns_from)
        return self.X_train, self.X_test
