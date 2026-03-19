from utils.ml_utils import generate_sample_dataset, build_and_train
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "food_spoilage_pipeline.pkl")

if __name__ == "__main__":
    print("Generating sample dataset and training pipeline...")
    df = generate_sample_dataset(csv_path=DATA_PATH, n_samples=1200)
    df.to_csv(DATA_PATH, index=False)
    summary = build_and_train(csv_path=DATA_PATH, model_output=MODEL_PATH)

    print("Training complete")
    print(f"Best model: {summary['best_model']}")
    print("Metrics:")
    for name, metrics in summary["metrics"].items():
        print(f"- {name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
    print(f"Saved model to: {summary['model_path']}")