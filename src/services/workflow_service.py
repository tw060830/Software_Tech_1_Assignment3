from pathlib import Path

import joblib
import pandas as pd

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, RAW_DATA_DIR, SUPPORTED_EXTENSIONS
from src.services.classifier_service import ClassifierService
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor


class WorkflowService:
    """Coordinate the shared workflow used by batch, GUI, and console entry points."""

    def __init__(self) -> None:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.preprocessor = ImagePreprocessor()
        self.classifier = ClassifierService(self.preprocessor, MODEL_OUTPUT_DIR)

        # Cached dataframe from the most recent index operation.
        self._dataframe: pd.DataFrame | None = None

    # Dataset discovery and indexing

    def discover_classes(self, data_dir: Path) -> list[str]:
        """
        Return a sorted list of class names found in *data_dir*.

        A class is any immediate sub-directory that contains at least one
        supported image file.

        Parameters
        ----------
        data_dir:
            Root folder of the dataset (contains one sub-folder per class).

        Returns
        -------
        list[str]
            Sorted class names (sub-directory names).
        """
        classes = set()
        for path in data_dir.iterdir():
            if path.is_dir():
                has_image = any(
                    f.suffix.lower() in SUPPORTED_EXTENSIONS
                    for f in path.rglob("*")
                    if f.is_file()
                )
                if has_image:
                    classes.add(path.name)
        return sorted(classes)

    def _build_dataframe(
            self,
            data_dir: Path | None = None,
            filter_classes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Index the dataset and optionally filter to a subset of classes.

        Parameters
        ----------
        data_dir:
            Dataset root. Defaults to RAW_DATA_DIR from config.
        filter_classes:
            If provided, only rows whose label appears in this list are
            kept.  Must contain at least 3 entries when supplied.

        Returns
        -------
        pd.DataFrame
            One row per image with columns: file_path, label, width,
            height, channels.

        Raises
        ------
        ValueError
            If *filter_classes* contains fewer than 3 entries.
        """
        if filter_classes is not None and len(filter_classes) < 3:
            raise ValueError(
                f"At least 3 classes must be selected; got {len(filter_classes)}."
            )

        indexer = DatasetIndexer(data_dir or RAW_DATA_DIR)
        dataframe = indexer.build_dataframe()

        if filter_classes is not None:
            dataframe = dataframe[dataframe["label"].isin(filter_classes)].copy()

        self._dataframe = dataframe
        return dataframe

    def load_dataframe(
            self,
            data_dir: Path | None = None,
            filter_classes: list[str] | None = None,
            force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Return the cached dataframe, re-indexing only when necessary.

        Parameters
        ----------
        data_dir:
            Dataset root. Triggers a reload if different from the cached
            dataset.
        filter_classes:
            Class filter forwarded to :meth:`_build_dataframe`.
        force_reload:
            When ``True``, always re-index even if a cache exists.
        """
        if self._dataframe is None or force_reload or filter_classes is not None:
            return self._build_dataframe(data_dir, filter_classes)
        return self._dataframe

    # Option 1 — Show dataset summary (filtered to chosen classes)

    def show_summary(
            self,
            data_dir: Path | None = None,
            filter_classes: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Build and return dataset summary statistics for the chosen classes.

        Parameters
        ----------
        data_dir:
            Dataset root folder.
        filter_classes:
            Subset of class names to include (minimum 3).

        Returns
        -------
        dict[str, float]
            Keys: total_images, total_classes, mean_width, mean_height,
            plus one count_<class> entry per selected class.
        """
        dataframe = self.load_dataframe(
            data_dir=data_dir,
            filter_classes=filter_classes,
            force_reload=True,
        )
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.build_summary()

        # Add per-class counts to the summary dict.
        class_counts = dataframe["label"].value_counts().to_dict()
        for class_name, count in sorted(class_counts.items()):
            summary[f"count_{class_name}"] = count

        return summary

    # Option 2 — Generate EDA outputs

    def generate_eda(
            self,
            data_dir: Path | None = None,
            filter_classes: list[str] | None = None,
    ) -> None:
        """Create and save the main EDA charts."""
        dataframe = self.load_dataframe(data_dir=data_dir, filter_classes=filter_classes)
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        eda.save_class_distribution()
        eda.save_image_size_distribution()

    # Option 3 — Train baseline classifier

    def train_model(
            self,
            data_dir: Path | None = None,
            filter_classes: list[str] | None = None,
    ) -> dict[str, object]:
        """Train the baseline model and save it to disk."""
        dataframe = self.load_dataframe(data_dir=data_dir, filter_classes=filter_classes)
        results = self.classifier.train(dataframe)
        self.classifier.save_model()
        return results

    # Option 4 — Predict image class

    def predict_image(self, file_path: str) -> str:
        """
        Predict the class of one input image.

        Raises
        ------
        FileNotFoundError
            If no trained model file exists.
        ValueError
            If the image cannot be read.
        """
        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                "No trained model found. Run Option 3 to train the classifier first."
            )
        self.classifier.model = joblib.load(model_path)
        features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = self.classifier.model.predict(features)[0]
        print(f"  Predicted class: {prediction}")

        if hasattr(self.classifier.model, "predict_proba"):
            probability = self.classifier.model.predict_proba(features).max()
            print(f"  Confidence     : {probability:.2%}")

        return str(prediction)

    # Option 5 — Export dataset index to CSV

    def export_index_to_csv(self, data_dir: Path) -> Path:
        """
        Index the entire dataset at *data_dir* and write it to a CSV file.

        The file is saved to ``outputs/reports/dataset_index.csv``.

        Parameters
        ----------
        data_dir:
            Dataset root folder (all classes are included).

        Returns
        -------
        Path
            Absolute path of the saved CSV file.
        """
        reports_dir = MODEL_OUTPUT_DIR.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        dataframe = self._build_dataframe(data_dir=data_dir)
        output_path = reports_dir / "dataset_index.csv"
        dataframe.to_csv(output_path, index=False)
        return output_path

    # Option 6 — Per-class image statistics

    def per_class_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame of per-class image dimension statistics.

        Uses the most recently loaded dataset index.  Call
        :meth:`load_dataframe` or :meth:`show_summary` first.

        Returns
        -------
        pd.DataFrame
            Columns: label, count, mean_width, mean_height, min_width,
            max_width, min_height, max_height.

        Raises
        ------
        RuntimeError
            If no dataset has been indexed yet.
        """
        if self._dataframe is None or self._dataframe.empty:
            raise RuntimeError(
                "No dataset indexed yet. Run Option 1 or Option 6 first."
            )

        grouped = (
            self._dataframe.groupby("label")
            .agg(
                count=("label", "count"),
                mean_width=("width", "mean"),
                mean_height=("height", "mean"),
                min_width=("width", "min"),
                max_width=("width", "max"),
                min_height=("height", "min"),
                max_height=("height", "max"),
            )
            .reset_index()
            .sort_values("label")
        )
        return grouped

    # Convenience — full pipeline (used by main.py)

    def run_full_pipeline(self) -> None:
        """Run the default Stage 1 and Stage 2 workflow non-interactively."""
        self.show_summary()
        self.generate_eda()
        results = self.train_model()
        print(f"Training accuracy: {results['accuracy']:.4f}")
        print(results["report"])