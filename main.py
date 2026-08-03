from Data.dataset import DatasetLoader
from Preprocessing.data_preprocessor import DataPreprocessor
from Models.ANN_model import ANNModel
from Training.trainer import ModelTrainer
from Evaluation.evaluator import ModelEvaluator
from Visualizations.plots import TrainingPlots
from Prediction.predict import WinePredictor

def main():

    file_path = "Data/winequalityN.csv"
    loader = DatasetLoader(file_path)
    dataset = loader.load_dataset()
    # shape
    loader.dataset_shape()
    # column
    loader.dataset_columns()
    # infomation
    loader.dataset_info()
    # checking missing value
    loader.missing_values()
    # cheking dupllicate reocrd
    loader.duplicate_values()
    # checking  head off dataset
    loader.first_five_rows()
    # statistical summary
    loader.statistical_summary()

    print("DATA PREPROCESSING")
    # Data Preprocessing
    preprocessor = DataPreprocessor(dataset)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    print("\n=== PREPROCESSING COMPLETED ===")
    print("Training Features Shape :", X_train.shape)
    print("Testing Features Shape  :", X_test.shape)
    print("Training Labels Shape   :", y_train.shape)
    print("Testing Labels Shape    :", y_test.shape)


 # BUILD ANN MODEL
    print("BUILDING ANN MODEL")
    print("=" * 60)
    ann = ANNModel(input_shape=X_train.shape[1])
    model = ann.build_model()
    ann.compile_model()
    ann.model_summary()

# MODEL TRAINING
    print("MODEL TRAINING")
    print("=" * 60)
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train)
    print("\nTraining Finished Successfully!")

    # model evaluation
    print("MODEL EVALUATION")
    print("=" * 60)
    evaluator = ModelEvaluator(model)
    evaluator.evaluate(
        X_test,
        y_test
)
    print("VISUALIZATION")
    plots = TrainingPlots(history)
    plots.plot_accuracy()
    plots.plot_loss()

# prediction

    predictor = WinePredictor("best_ann_model.keras")

    sample = [
        7.0,      # fixed acidity
        0.27,     # volatile acidity
        0.36,     # citric acid
        20.7,     # residual sugar
        0.045,    # chlorides
        45.0,     # free sulfur dioxide
        170.0,    # total sulfur dioxide
        1.0010,   # density
        3.00,     # pH
        0.45,     # sulphates
        8.8       # alcohol
    ]

    predictor.predict(sample)

if __name__ == "__main__":
    main()