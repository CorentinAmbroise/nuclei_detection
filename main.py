import fire

from scripts import (
    explore_dataset,
    generate_mask,
    train_model,
    evaluate_model,
)

if __name__ == "__main__":
    fire.Fire({
        "explore-dataset": explore_dataset.explore_dataset,
        "train-model": train_model.train_classification_model,
        "train-model-with-regularization": train_model.train_with_anti_overfitting,
        "evaluate-model": evaluate_model.evaluate_saved_model,
        "generate-nuclei-mask": generate_mask.generated_nuclei_mask,
        "display-wsi-with-mask": generate_mask.display_wsi_with_mask,
    })
