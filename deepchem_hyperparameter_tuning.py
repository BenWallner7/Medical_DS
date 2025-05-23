## Imports

import numpy as np

import os
import pickle
from dotenv import load_dotenv

import shutil

import optuna
import deepchem as dc
from deepchem.models import AttentiveFPModel
from deepchem.metrics import Metric, roc_auc_score

# Built in upsampling

import torch

from sklearn.metrics import roc_auc_score


from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

import json

# Supress RDKit warnings

from rdkit import RDLogger

logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


# Set random seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Env variables

load_dotenv("/home/raid/bw507/JHR_Research_Project_Repo/env_files/.env")

## Load Data

#Standard split

# with open(
#     "/home/raid/bw507/JHR_Research_Project_Repo/graph_neural_networks/featurized_data/balanced_training_dataset.pkl",
#     "rb",
# ) as f:
#     balanced_train_dataset = pickle.load(f)

# Similar structures removed split
with open(
    "/home/raid/bw507/JHR_Large_Files/graph_inputs/balanced_training_dataset_similar_structures_removed.pkl",
    "rb",
) as f:
    balanced_train_dataset = pickle.load(f)

print(balanced_train_dataset)
## Early Stopping


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


class EarlyStoppingCV:
    def __init__(
        self, model, patience=10, min_delta=0.0, mode="max", checkpoint_dir=None
    ):
        """
        Args:
            model (dc.models.Model): DeepChem model to monitor.
            patience (int): Number of epochs to wait after no improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'max' if higher metric is better (e.g., AUC), 'min' for loss.
            checkpoint_dir (str): Directory to save the model checkpoint.
        """
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir or model.model_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self._save_checkpoint(epoch)
            return

        improvement = (
            (current_score - self.best_score)
            if self.mode == "max"
            else (self.best_score - current_score)
        )

        if improvement > self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            print(
                f"Improved score to {current_score:.4f} at epoch {epoch}. Saving model..."
            )
            self._save_checkpoint(epoch)
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True

    def _save_checkpoint(self, epoch):
        """
        Save a checkpoint of the model.
        Uses a temporary file and disables new zipfile serialization to avoid known issues.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        data = {
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.model._pytorch_optimizer.state_dict(),
            "global_step": self.model._global_step,
        }
        temp_path = os.path.join(
            self.checkpoint_dir, f"temp_checkpoint_epoch_{epoch}.pt"
        )
        try:
            torch.save(data, temp_path, _use_new_zipfile_serialization=False)
            shutil.move(temp_path, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except RuntimeError as e:
            print(f"Error saving checkpoint at epoch {epoch}: {e}")


## Splitting


def get_subset(dataset, indices):
    X_sub = dataset.X[indices]
    y_sub = dataset.y[indices]
    w_sub = dataset.w[indices]
    ids_sub = dataset.ids[indices]
    return dc.data.NumpyDataset(X_sub, y_sub, w_sub, ids_sub)


## Train on Fold


def train_on_fold(train_data, val_data, fold_model_dir, params):
    """
    Train an AttentiveFP model on one fold with given hyperparameters.
    Returns best validation AUC and the trained model.
    """
    # Create model for this fold:
    model = dc.models.AttentiveFPModel(
        n_tasks=1,
        mode="classification",
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        dropout=params["dropout"],
        num_layers=params["num_layers"],
        graph_feat_size=params["graph_feat_size"],
        num_timesteps=params["num_timesteps"],
        number_atom_features=33,
        model_dir=fold_model_dir,
    )

    metric = Metric(roc_auc_score, mode="classification")
    # Use a ValidationCallback to log validation performance (does not enforce early stopping by itself)
    callbacks = [dc.models.ValidationCallback(val_data, 1000, metric)]
    early_stopper = EarlyStoppingCV(
        model=model,
        patience=10,
        min_delta=0.0001,
        mode="max",
        checkpoint_dir=fold_model_dir,
    )

    best_fold_auc = -np.inf
    for epoch in range(50):
        model.fit(train_data, nb_epoch=1, callbacks=callbacks, checkpoint_interval=10)
        val_scores = model.evaluate(val_data, [metric])
        val_auc = val_scores["roc_auc_score"]
        print(f"Fold: {fold_model_dir}, Epoch {epoch+1}: Val AUC: {val_auc:.4f}")
        early_stopper(val_auc, epoch)
        if val_auc > best_fold_auc:
            best_fold_auc = val_auc
        if early_stopper.early_stop:
            break

    return best_fold_auc, model


def objective(trial):

    print(f"Starting trial {trial.number}...")

    # Define hyperparameters to tune:
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.9),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "graph_feat_size": trial.suggest_int("graph_feat_size", 100, 600),
        "num_timesteps": trial.suggest_int("num_timesteps", 1, 8),
        # "number_atom_features": trial.suggest_int("number_atom_features", 20, 150),
        # "number_bond_features": trial.suggest_int("number_bond_features", 5, 75),
    }

    # Use stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y = balanced_train_dataset.y.ravel()
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n========== Fold {fold+1}/3 ==========")
        train_data = get_subset(balanced_train_dataset, train_idx)
        val_data = get_subset(balanced_train_dataset, val_idx)
        fold_model_dir = (
            f"/home/raid/bw507/JHR_Large_Files/model_builds/attentivefp_model_fold_similar_structures_removed{fold+1}"
        )
        # Remove existing folder if present:
        if os.path.exists(fold_model_dir):
            shutil.rmtree(fold_model_dir, ignore_errors=True)

        fold_auc, _ = train_on_fold(train_data, val_data, fold_model_dir, params)
        fold_aucs.append(fold_auc)

    avg_auc = np.mean(fold_aucs)
    print(f"\nAverage Validation AUC across folds: {avg_auc:.4f}")
    
    # We want to maximize AUC, so return it
    
    # Print details of this trial to terminal
    print(f"Trial {trial.number}:")
    print(f"  Parameters: {trial.params}")
    print(f"  PR AUC Scores: {fold_aucs}")
    print(f"  Mean PR AUC: {avg_auc}")
    print("-" * 50)

    # Also update the JSON file with this trial's results
    trial_data = {
        "number": trial.number,
        "params": trial.params,
        "value": float(avg_auc),
        "auc_scores": [float(score) for score in fold_aucs],
        "user_attrs": trial.user_attrs,
    }

    # Load existing results if file exists
    try:
        with open("trial_results_attentivefp_tuning_similar_structure_split.json", "r") as f:
            all_trials = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_trials = []

    # Add current trial and save
    all_trials.append(trial_data)
    with open("trial_results_attentivefp_tuning_similar_structure_split.json", "w") as f:
        json.dump(all_trials, f, indent=2)

    return avg_auc


# Run Optuna optimization:
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Retrieve best hyperparameters
print("Best hyperparameters:", study.best_trial.params)
best_params = study.best_trial.params

with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best parameters saved to `best_params.json`")

# After hyperparameter tuning, retrain on the full dataset using the best params

final_model_dir = "/home/raid/bw507/JHR_Large_Files/model_builds/attentivefp_model_final_similar_structures_removed"
if os.path.exists(final_model_dir):
    shutil.rmtree(final_model_dir, ignore_errors=True)
final_model = AttentiveFPModel(
    n_tasks=1,
    mode="classification",
    batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    dropout=best_params["dropout"],
    num_layers=best_params["num_layers"],
    graph_feat_size=best_params["graph_feat_size"],
    num_timesteps=best_params["num_timesteps"],
    number_atom_features=33,
    model_dir=final_model_dir,
)


# Re-train best model parameters on full dataset

splitter = dc.splits.RandomSplitter()


train_dataset, validation_dataset, test_dataset = splitter.train_valid_test_split(
  dataset=balanced_train_dataset, frac_train=0.8, frac_valid=0.2, frac_test=0.0
)
   

metric = Metric(roc_auc_score, mode="classification")
# Use a ValidationCallback to log validation performance (does not enforce early stopping by itself)
callbacks = [dc.models.ValidationCallback(validation_dataset, 1000, metric)]
early_stopper = EarlyStopping(patience=10, min_delta=0.0001, mode="max")

# Train for up to 100 epochs
losses = []

test_losses = []

train_scores_list = []

test_scores_list = []

for i in range(250):
    loss = final_model.fit(
        train_dataset, nb_epoch=1, callbacks=callbacks, checkpoint_interval=10
    )
    losses.append(loss)

    # Calculate metrics every 5 epochs
    if (i + 1) % 5 == 0:
        train_scores = final_model.evaluate(train_dataset, [metric])
        test_scores = final_model.evaluate(validation_dataset, [metric])

        train_scores_list.append(train_scores)
        test_scores_list.append(test_scores)

        test_auc = test_scores["roc_auc_score"]

        print(
            f"Epoch {i + 1}: Train AUC: {train_scores['roc_auc_score']:.4f}, Test AUC: {test_auc:.4f}"
        )

        # Evaluate on validation data
        # test_loss = final_model.evaluate(validation_dataset, [dc.metrics.Metric(dc.metrics.mean_squared_error)])
        # test_losses.append(test_loss)

        # Implement early stopping based on validation AUC
        early_stopper(test_auc, i)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {i + 1}")
            break


final_model.save_checkpoint()
print("Final model trained and saved.")

# Collect all trial results
trial_results = []
for t in study.trials:
    trial_results.append(
        {
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "user_attrs": t.user_attrs,
        }
    )

# Save the trial results to a JSON file
with open("trial_results.json", "w") as f:
    json.dump(trial_results, f, indent=4)

print("Trial results saved to trial_results.json")
