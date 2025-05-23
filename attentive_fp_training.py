import os

os.environ["TMPDIR"] = "/local/data/public/bw507/custom_tmp"
os.environ["TEMP"] = "/local/data/public/bw507/custom_tmp"
os.environ["TMP"] = "/local/data/public/bw507/custom_tmp"
os.environ["DEEPCHEM_DATA_DIR"] = "/local/data/public/bw507/custom_tmp"

from deepchem.models.torch_models import AttentiveFPModel
import deepchem as dc
from deepchem.metrics import (
    Metric,
    roc_auc_score,
    prc_auc_score,
    balanced_accuracy_score,
)


import tensorflow as tf

import numpy as np
import pickle
import os


# Set random seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Similar structure split

with open(
    "/home/raid/bw507/JHR_Large_Files/graph_inputs/balanced_training_dataset_similar_structures_removed.pkl",
    "rb",
) as f:
    balanced_dataset = pickle.load(f)

with open(
    "/home/raid/bw507/JHR_Large_Files/graph_inputs/testing_dataset_similar_structures_removed.pkl",
    "rb",
) as f:
    test_dataset = pickle.load(f)


# Random Activity Split

# with open(
#     "/home/raid/bw507/JHR_Large_Files/graph_inputs/balanced_training_dataset.pkl",
#     "rb",
# ) as f:
#     balanced_dataset = pickle.load(f)

# For CV splits

splitter = dc.splits.RandomSplitter()
# folds = splitter.k_fold_split(balanced_dataset, k=5)

# train_dataset, validation_dataset, test_dataset = splitter.train_valid_test_split(
#   dataset=balanced_dataset, frac_train=0.8, frac_valid=0.2, frac_test=0.0
# )

train_folds = splitter.k_fold_split(balanced_dataset, k=5)
val_folds = splitter.k_fold_split(test_dataset, k=5)


def get_subset(dataset, indices):
    X_sub = dataset.X[indices]
    y_sub = dataset.y[indices]
    w_sub = dataset.w[indices]
    ids_sub = dataset.ids[indices]
    return dc.data.NumpyDataset(X_sub, y_sub, w_sub, ids_sub)


# Early Stopping


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
        """
        Args:
            patience (int): Number of epochs to wait after no improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'max' for metrics like AUC, 'min' for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return

        improvement = (
            (current_score - self.best_score)
            if self.mode == "max"
            else (self.best_score - current_score)
        )

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch}")
        self.model.save_checkpoint(max_checkpoints_to_keep=1)


# Enrichment factor custom metric


def enrichment_factor(y_true, y_scores, needs_proba=True, top_fraction=0.05):
    """
    Calculate the Enrichment Factor (EF) for a set of predictions.

    Parameters:
    y_true (array-like): Binary labels indicating active (1) or inactive (0) compounds.
    y_scores (array-like): Predicted scores or probabilities from the model.
    top_fraction (float): Fraction of the dataset to consider for EF calculation (e.g., 0.01 for top 1%).

    Returns:
    float: Enrichment Factor.
    """
    y_true = np.array(np.argmax(y_true, axis=1))
    y_scores = np.array(y_scores[:, 1])
    # print(y_true)
    # print(y_scores)
    N = len(y_true)
    N_active = np.sum(y_true)
    top_n = int(np.ceil(N * top_fraction))
    sorted_indices = np.argsort(y_scores)[::-1]  # Descending sort
    top_indices = sorted_indices[:top_n]
    hits_top = np.sum(y_true[top_indices])
    print(f"N={N}, N_active={N_active}, top_n={top_n}, hits_top={hits_top}")
    ef = (hits_top / top_n) / (N_active / N)
    return ef


ef_metric = Metric(
    metric=enrichment_factor,
    name="enrichment_factor_score",
    mode="classification",
)

# Collect all metrics

pr_auc = Metric(prc_auc_score, mode="classification", name="pr_auc_score")

roc_auc = Metric(roc_auc_score, name="roc_auc_score", mode="classification")

balanced_accuracy = Metric(balanced_accuracy_score, name="balanced_accuracy_score")


metrics = [pr_auc, balanced_accuracy, roc_auc, ef_metric]


# Main training loop

final_model_dir = "/home/raid/bw507/JHR_Large_Files/model_builds/attentivefp_model_cross_validated_final_structure_similarity_split"

roc_auc_scores = []
pr_auc_scores = []
balanced_accuracy_scores = []
enrichment_factor_scores = []

# for fold_idx, (train_ds, valid_ds) in enumerate(folds):
for fold_idx in range(5):
    train_ds = train_folds[fold_idx][0]
    valid_ds = val_folds[fold_idx][0]
    print(train_ds)
    print(valid_ds)
    print(f"\n=== Fold {fold_idx+1} ===")

    callbacks = dc.models.ValidationCallback(
        dataset=valid_ds,
        interval=1000,
        metrics=metrics,
        save_metric=0,
        save_on_minimum=False,
    )

    early_stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="max")

    model = AttentiveFPModel(
        n_tasks=1,
        mode="classification",
        batch_size=32,
        learning_rate=0.0001506094804554213,
        dropout=0.6782305519626475,
        num_layers=3,
        graph_feat_size=297,
        num_timesteps=7,
        number_atom_features=33,
        model_dir=final_model_dir + "_fold_{}".format(fold_idx + 1),
    )

    # callbacks = [ValidationCallback(valid_ds, 1000, metric), early_stopper]

    # Epoch loop
    for epoch in range(300):
        _ = model.fit(train_ds, nb_epoch=1, checkpoint_interval=5)
        train_scores = model.evaluate(dataset=train_ds, metrics=metrics)
        valid_scores = model.evaluate(dataset=valid_ds, metrics=metrics)
        print(
            f" Epoch {epoch+1}: Train ROC AUC = {train_scores['roc_auc_score']:.4f}, "
            f"Valid ROC AUC = {valid_scores['roc_auc_score']:.4f}"
        )
        print(
            f" Epoch {epoch+1}: Train PR AUC = {train_scores['pr_auc_score']:.4f}, "
            f"Valid PR AUC = {valid_scores['pr_auc_score']:.4f}"
        )
        print(
            f" Epoch {epoch+1}: Train Balanced Accuracy = {train_scores['balanced_accuracy_score']:.4f}, "
            f"Valid Balanced Accuracy = {valid_scores['balanced_accuracy_score']:.4f}"
        )
        print(
            f" Epoch {epoch+1}: Train Enrichment Factor = {train_scores['enrichment_factor_score']:.4f}, "
            f"Valid Enrichment Factor = {valid_scores['enrichment_factor_score']:.4f}"
        )

        early_stopper(valid_scores["enrichment_factor_score"], epoch)

        if early_stopper.early_stop:
            print(f"Stopping at epoch {epoch+1}")
            break

    # Record fold result
    final_valid = model.evaluate(valid_ds, metrics=metrics)
    roc_auc_scores.append(final_valid["roc_auc_score"])
    pr_auc_scores.append(final_valid["pr_auc_score"])
    balanced_accuracy_scores.append(final_valid["balanced_accuracy_score"])
    enrichment_factor_scores.append(final_valid["enrichment_factor_score"])

print(f"\nCV ROC AUC: {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
print(f"\nCV PR AUC: {np.mean(pr_auc_scores):.4f} ± {np.std(pr_auc_scores):.4f}")
print(
    f"\nCV Balanced Accuracy: {np.mean(balanced_accuracy_scores):.4f} ± {np.std(balanced_accuracy_scores):.4f}"
)
print(
    f"\nCV Enrichment Factor: {np.mean(enrichment_factor_scores):.4f} ± {np.std(enrichment_factor_scores):.4f}"
)

# final_model = AttentiveFPModel(
#     n_tasks=1,
#     mode="classification",
#     batch_size=32,
#     learning_rate=0.0001506094804554213,
#     dropout=0.6782305519626475,
#     num_layers=3,
#     graph_feat_size=297,
#     num_timesteps=7,
#     number_atom_features=33,
#     model_dir=final_model_dir,
# )


# losses = []

# test_losses = []

# train_scores_list = []

# test_scores_list = []

# for i in range(500):

#     print("Starting Epoch {}".format(i))
#     loss = final_model.fit(
#         train_dataset, nb_epoch=1, callbacks=callbacks, checkpoint_interval=10
#     )
#     losses.append(loss)

#     # Calculate metrics every 5 epochs
#     if (i + 1) % 5 == 0:
#         train_scores = final_model.evaluate(train_dataset, [metric])
#         test_scores = final_model.evaluate(validation_dataset, [metric])

#         train_scores_list.append(train_scores)
#         test_scores_list.append(test_scores)

#         test_auc = test_scores["roc_auc_score"]

#         print(
#             f"Epoch {i + 1}: Train AUC: {train_scores['roc_auc_score']:.4f}, Test AUC: {test_auc:.4f}"
#         )

#         # Evaluate on validation data
#         # test_loss = final_model.evaluate(validation_dataset, [dc.metrics.Metric(dc.metrics.mean_squared_error)])
#         # test_losses.append(test_loss)

#         # Implement early stopping based on validation AUC
#         early_stopper(test_auc, i)
#         if early_stopper.early_stop:
#             print(f"Early stopping triggered at epoch {i + 1}")
#             break


# final_model.save_checkpoint()
# print("Final model trained and saved.")
