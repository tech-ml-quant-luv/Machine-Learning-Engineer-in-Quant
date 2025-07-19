import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
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

    def load_data(self, file_path, delimiter=',', target_column=-1):
        self.dataset = pd.read_csv(file_path, delimiter=delimiter)
        self.X = self.dataset.iloc[:, :-1].values
        self.Y = self.dataset.iloc[:, target_column].values
        print("[INFO] Data loaded successfully.")
        return self.X, self.Y

    def handle_missing_data(self, impute_columns=(1, 2), strategy='mean'):
        self.imputer = SimpleImputer(missing_values=float("nan"), strategy=strategy)
        start, end = impute_columns
        self.X[:, start:end + 1] = self.imputer.fit_transform(self.X[:, start:end + 1])
        print(f"[INFO] Missing values handled using '{strategy}' strategy on columns {start} to {end}.")
        return self.X

    def encode_features(self, onehot_column=0):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), [onehot_column])
            ],
            remainder='passthrough'
        )
        self.X = self.column_transformer.fit_transform(self.X)
        print(f"[INFO] One-hot encoding applied to column {onehot_column}.")
        return self.X

    def encode_target(self, apply_encoding=True):
        if apply_encoding:
            self.Y = self.label_encoder_Y.fit_transform(self.Y)
            print("[INFO] Target variable label-encoded.")
        else:
            print("[INFO] Target encoding skipped.")
        return self.Y

    def split_data(self, test_size=0.2, random_state=1):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        print(f"[INFO] Data split into training and test sets with test_size={test_size}, random_state={random_state}.")
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def scale_features(self, scale_columns_from=1):
        self.X_train[:, scale_columns_from:] = self.scaler.fit_transform(self.X_train[:, scale_columns_from:])
        self.X_test[:, scale_columns_from:] = self.scaler.transform(self.X_test[:, scale_columns_from:])
        print(f"[INFO] Features scaled from column index {scale_columns_from}.")
        return self.X_train, self.X_test
