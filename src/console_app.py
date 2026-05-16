from pathlib import Path

from src.services.workflow_service import WorkflowService

class ConsoleApp:
    """Menu-driven console application for the assignment workflow."""

    def __init__(self, workflow_service: WorkflowService) -> None:
        self.workflow_service = workflow_service

    def _prompt_dataset_path(self) -> Path:
        """Ask the user for a dataset root directory and validate it."""
        while True:
            raw = input("Enter the path to your dataset folder: ").strip()
            if not raw:
                print("  Path cannot be empty. Please try again.")
                continue
            path = Path(raw)
            if not path.exists():
                print(f"  Path not found: {path}. Please try again.")
                continue
            if not path.is_dir():
                print(f"  That path is not a directory. Please try again.")
                continue
            return path

    def _select_classes(
        self,
        available: list[str],
        minimum: int = 3,
    ) -> list[str]:
        """
        Display available classes and let the user choose a subset.

        Parameters
        ----------
        available:
            All class names found in the dataset.
        minimum:
            Minimum number of classes the user must select.

        Returns
        -------
        list[str]
            The class names chosen by the user.
        """
        print(f"\n  Found {len(available)} classes:")
        for idx, name in enumerate(available, start=1):
            print(f"    {idx:>3}. {name}")

        print(
            f"\n  Enter the numbers of at least {minimum} classes to include,"
            " separated by commas."
        )
        print("  Example: 1,3,5")

        while True:
            raw = input("  Your selection: ").strip()
            try:
                indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                print("  Please enter numbers separated by commas.")
                continue

            valid = [i for i in indices if 1 <= i <= len(available)]
            unique_valid = list(dict.fromkeys(valid))  # preserve order, dedupe

            if len(unique_valid) < minimum:
                print(
                    f"  You must select at least {minimum} classes."
                    f" You selected {len(unique_valid)} valid choice(s)."
                )
                continue

            chosen = [available[i - 1] for i in unique_valid]
            print(f"\n  Selected classes: {', '.join(chosen)}")
            return chosen

    def _action_show_summary(self) -> None:
        """
        Option 1 — Show dataset summary.

        Asks the user for a folder path and which classes (minimum 3) to
        include, then prints summary statistics for only those classes.
        """
        print("\n--- Show Dataset Summary ---")
        dataset_path = self._prompt_dataset_path()

        # Discover available classes (immediate subdirectories that contain
        # at least one supported image file).
        available_classes = self.workflow_service.discover_classes(dataset_path)
        if len(available_classes) < 3:
            print(
                "  The selected folder contains fewer than 3 recognisable"
                " class sub-folders. Please choose a different dataset path."
            )
            return

        chosen_classes = self._select_classes(available_classes, minimum=3)
        summary = self.workflow_service.show_summary(
            data_dir=dataset_path,
            filter_classes=chosen_classes,
        )

        print("\n  === Dataset Summary ===")
        for key, value in summary.items():
            label = key.replace("_", " ").title()
            print(f"  {label}: {value}")

    def _action_generate_eda(self) -> None:
        """Option 2 — Generate and save EDA charts."""
        print("\n--- Generate EDA Outputs ---")
        self.workflow_service.generate_eda()
        print("  EDA outputs saved to outputs/eda/")

    def _action_train_model(self) -> None:
        """Option 3 — Train the baseline classifier."""
        print("\n--- Train Baseline Classifier ---")
        results = self.workflow_service.train_model()
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print("\n  Classification Report:")
        print(results["report"])

    def _action_predict_image(self) -> None:
        """Option 4 — Predict the class of a single image."""
        print("\n--- Predict Image Class ---")
        image_path = input("  Enter image path: ").strip()
        try:
            self.workflow_service.predict_image(image_path)
        except FileNotFoundError as error:
            print(f"  Error: {error}")
        except ValueError as error:
            print(f"  Error: {error}")

    def _action_export_index(self) -> None:
        """
        Option 5 — Export the dataset index to a CSV file.

        Asks for a dataset path, builds the full index for all classes,
        and writes the result to outputs/reports/dataset_index.csv.
        """
        print("\n--- Export Dataset Index to CSV ---")
        dataset_path = self._prompt_dataset_path()
        output_path = self.workflow_service.export_index_to_csv(dataset_path)
        print(f"  Index exported to: {output_path}")

    def _action_per_class_stats(self) -> None:
        """
        Option 6 — Show per-class image dimension statistics.

        Displays a table of min, max, and mean width/height for each
        class found in the currently loaded dataset index.
        """
        print("\n--- Per-Class Image Statistics ---")
        try:
            stats = self.workflow_service.per_class_stats()
            if stats.empty:
                print("  No data available. Run Option 1 or 6 first to index the dataset.")
                return
            print(f"\n  {'Class':<30} {'Count':>6} {'Mean W':>8} {'Mean H':>8}"
                  f" {'Min W':>7} {'Max W':>7} {'Min H':>7} {'Max H':>7}")
            print("  " + "-" * 88)
            for _, row in stats.iterrows():
                print(
                    f"  {row['label']:<30} {int(row['count']):>6}"
                    f" {row['mean_width']:>8.1f} {row['mean_height']:>8.1f}"
                    f" {int(row['min_width']):>7} {int(row['max_width']):>7}"
                    f" {int(row['min_height']):>7} {int(row['max_height']):>7}"
                )
        except RuntimeError as error:
            print(f"  Error: {error}")

    def run(self) -> None:
        """Start the menu loop until the user chooses to exit."""
        menu = (
            "\n  Macroinvertebrate Image Analysis System",
            "  " + "=" * 42,
            "   1. Show dataset summary (choose classes)",
            "   2. Generate EDA outputs",
            "   3. Train baseline classifier",
            "   4. Predict an image",
            "   5. Export dataset index to CSV",
            "   6. Show per-class image statistics",
            "   7. Exit",
            "  " + "=" * 42,
        )

        actions = {
            "1": self._action_show_summary,
            "2": self._action_generate_eda,
            "3": self._action_train_model,
            "4": self._action_predict_image,
            "5": self._action_export_index,
            "6": self._action_per_class_stats,
        }

        while True:
            print("\n".join(menu))
            choice = input("  Select an option: ").strip()

            if choice == "7":
                print("\n  Exiting application. Goodbye.\n")
                break

            action = actions.get(choice)
            if action is None:
                print("  Invalid option. Please enter a number from the menu.")
            else:
                action()

def main() -> None:
    """Start the menu-driven console version of the project."""
    workflow = WorkflowService()
    app = ConsoleApp(workflow)
    app.run()

if __name__ == "__main__":
    main()