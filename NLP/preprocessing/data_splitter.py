from sklearn.model_selection import train_test_split


class DataSplitter:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        print("\n== TRAIN TEST SPLIT COMPLETED ===")
        print("Training Features :", X_train.shape)
        print("Testing Features  :", X_test.shape)
        print("Training Labels   :", y_train.shape)
        print("Testing Labels    :", y_test.shape)

        return X_train, X_test, y_train, y_test