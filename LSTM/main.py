from Preprocessing.dataset import DatasetLoader
from Preprocessing.text_preprocessing import TextPreprocessor
from Models.LSTM_model import LSTMModel
from Training.trainer import Trainer
from Evaluation.Evaluator import Evaluator
from Prediction.predictor import Predictor
def main():

    dataset = DatasetLoader("Data/IMDB Dataset.csv")
    X_train, X_test, y_train, y_test = dataset.load_dataset()

    # PREPROCESSING
    processor = TextPreprocessor()
    X_train, X_test, tokenizer = processor.preprocess(
        X_train,
        X_test
    )
    print("\nPreprocessing Completed Successfully!")

#       LSTM MODEL
    model_builder = LSTMModel()
    model = model_builder.build_model()

#     trainer
    trainer = Trainer(model)
    history = trainer.train(
        X_train,
        y_train)

#     evaluation

    evaluator = Evaluator(model)

    evaluator.evaluate(
        X_test,
        y_test
)

#     predictio
    predictor = Predictor(model, tokenizer)
    predictor.predict(
        "This movie was absolutely amazing. I loved every scene."
    )

if __name__ == "__main__":
    main()