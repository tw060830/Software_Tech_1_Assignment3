from pathlib import Path

import joblib
import pandas as pd

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR
from src.services.classifier_service import ClassifierService
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor

class WorkflowService:
    """Coordinate the shared workflow used by batch, GUI, and console entry points."""

    def __init__(self) -> None:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.classifier = ClassifierService(self.preprocessor, MODEL_OUTPUT_DIR)
        self.dataframe: pd.DataFrame | None = None

    def load_dataframe(self) -> pd.DataFrame:
        """Load and cache the indexed dataset."""

        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
        return self.dataframe

    def show_summary(self) -> dict[str, float]:
        """Build and print dataset summary statistics."""

        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.build_summary()
        print(summary)
        return summary

    def generate_eda(self) -> None:
        """Create and save the main EDA outputs."""

        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        eda.save_class_distribution()
        eda.save_image_size_distribution()

    def train_model(self) -> dict[str, object]:
        """Train the baseline model and save it to disk."""

        dataframe = self.load_dataframe()
        results = self.classifier.train(dataframe)
        self.classifier.save_model()
        return results
    def predict_image(self, file_path: str) -> str:
        """Predict the class of one input image."""

        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
        if model_path.exists():
            self.classifier.model = joblib.load(model_path)

        features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = self.classifier.model.predict(features)[0]
        print(f"Predicted class: {prediction}")
        return str(prediction)

    def run_full_pipeline(self) -> None:
        """Run the default Stage 1 and Stage 2 workflow."""

        self.show_summary()
        self.generate_eda()
        results = self.train_model()
        print(f"Training accuracy: {results['accuracy']:.4f}")
        print(results["report"])