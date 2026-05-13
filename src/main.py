from src.services.workflow_service import WorkflowService

def main() -> None:
    """Run the default non-interactive project workflow."""

    workflow = WorkflowService()
    workflow.run_full_pipeline()

if __name__ == "__main__":
    main()