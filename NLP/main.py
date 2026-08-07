from preprocessing.dataset import DatasetLoader
from preprocessing.text_preprocessor import TextPreprocessor
from Visualizations.Visualizer import Visualizer
from preprocessing.feature_extrator import FeatureExtractor
from preprocessing.data_splitter import DataSplitter
from Models.resume_classifier import ResumeClassifier
from Evaluation.evaluator import Evaluator
from Training.trainer import ModelTrainer
from Prediction.predictor import ResumePredictor

def main():
    file_path = "Data/Resume Screening.csv"
    loader = DatasetLoader(file_path)
    dataset = loader.load_dataset()
    loader.dataset_shape()
    loader.dataset_columns()
    loader.dataset_info()
    loader.missing_values()
    loader.duplicate_records()
    loader.first_five_rows()
    loader.category_distribution()

 # Text Preprocessing
    preprocessor = TextPreprocessor(dataset)
    preprocessor.check_missing_values()
    preprocessor.check_duplicates()
    dataset = preprocessor.clean_text()
    print("\n===== PREPROCESSED DATA ====")
    print(dataset.head())

    # Data Visualization

    visualizer = Visualizer(dataset)
    visualizer.plot_category_distribution()
    visualizer.word_cloud()

    # Feature Extraction
    extractor = FeatureExtractor(dataset)
    X, y, vectorizer, encoder = extractor.extract_features()

    # Train Test Split
    splitter = DataSplitter(X, y)
    X_train, X_test, y_train, y_test = splitter.split_data()

    # Model Training
    classifier = ResumeClassifier()
    model = classifier.train(X_train, y_train)

    # Model Evaluation
    evaluation = Evaluator(
    model,
    X_test,
    y_test
)
    evaluation.evaluate()

    # Save Model
    trainer = ModelTrainer(model, vectorizer)
    trainer.save_model()

# Resume Prediction

    predictor = ResumePredictor()
    predictor.predict_resume()

if __name__ == "__main__":
    main()