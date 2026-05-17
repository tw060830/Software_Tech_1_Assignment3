from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class ClassifierService:
    """Train, evaluate, and persist the baseline classification model."""

    def __init__(self, preprocessor, model_output_dir: Path) -> None:
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    def prepare_features(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert indexed file paths into model features and labels."""

        features = []
        labels = []

        for _, row in dataframe.iterrows():
            features.append(self.preprocessor.transform(row["file_path"]))
            labels.append(row["label"])

        return np.array(features), np.array(labels)

    def train(self, dataframe: pd.DataFrame) -> dict[str, object]:
        """Fit the model and return evaluation outputs."""

        X, y = self.prepare_features(dataframe)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        # Generate visualisations
        self._save_confusion_matrix(y_test, predictions)
        self._save_classification_report(y_test, predictions)

        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
        return results

    def _save_confusion_matrix(self, y_test: np.ndarray, predictions: np.ndarray) -> Path:
        """Generate and save confusion matrix visualization."""
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test), cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Macroinvertebrate Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = self.model_output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✓ Confusion matrix saved to {output_path}")
        return output_path

    def _save_classification_report(self, y_test: np.ndarray, predictions: np.ndarray) -> Path:
        """Generate and save classification report visualization."""
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        # Extract metrics for each class (excluding micro/macro/weighted averages)
        class_labels = sorted(
            [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']])

        precision_scores = [report[label]['precision'] for label in class_labels]
        recall_scores = [report[label]['recall'] for label in class_labels]
        f1_scores = [report[label]['f1-score'] for label in class_labels]

        # Create bar plot
        x = np.arange(len(class_labels))
        width = 0.25

        plt.figure(figsize=(16, 6))
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Classification Report - Precision, Recall, and F1-Score by Class')
        plt.xticks(x, class_labels, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = self.model_output_dir / 'classification_report.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✓ Classification report saved to {output_path}")
        return output_path

    def save_model(self, file_name="macro_classifier.joblib") -> Path:
        output_path = self.model_output_dir / file_name
        joblib.dump(self.model, output_path)
        print(f"✓ Model saved to {output_path}")
        return output_path