"""Question 4: Bagging ensemble of 3 base models -- kNN (knn.py),
IRT (item_response.py), and the Autoencoder (neural_network.py).

How the ensemble works:
  1. Bootstrap the training set 3 times (sample len(train_data) rows WITH
     replacement, once per base model).
  2. Train one base model on each bootstrap sample: kNN (user-based
     KNNImputer), IRT (theta/beta via gradient ascent), and the
     Autoencoder (PyTorch).
  3. For every (user_id, question_id) in valid_data / test_data, get a
     predicted probability of "correct" from each of the 3 trained
     models, and AVERAGE the 3 probabilities.
  4. Threshold the averaged probability at 0.5 to get the final
     prediction, and report validation/test accuracy.
"""

import numpy as np
import torch
from sklearn.impute import KNNImputer

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
from item_response import irt, sigmoid as irt_sigmoid
from neural_network import AutoEncoder, train as nn_train

BEST_KNN_K = 11
BEST_IRT_LR = 0.01
BEST_IRT_ITERATIONS = 150
NN_K = 10
NN_LR = 0.03
NN_LAMB = 0.001
NN_NUM_EPOCH = 20


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------
def bootstrap_matrix(train_data, matrix_shape, rng):
    """Bootstrap-resample the training observations into a
    (num_student, num_question) matrix with NaN for cells NOT selected by
    this draw. Used for kNN and the Autoencoder, which both expect a
    matrix, not a list. See module docstring for the duplicate-handling
    caveat.
    """
    n = len(train_data["user_id"])
    idx = rng.choice(n, size=n, replace=True)
    mat = np.full(matrix_shape, np.nan)
    for i in idx:
        u = train_data["user_id"][i]
        q = train_data["question_id"][i]
        c = train_data["is_correct"][i]
        mat[u, q] = c
    return mat


def bootstrap_dict(train_data, rng):
    """Bootstrap-resample the training observations, KEEPING duplicates,
    as a {user_id, question_id, is_correct} dict -- used for IRT, which
    operates on the list of observations directly (so duplicates really
    do get double-counted during gradient updates, unlike the matrix
    version above).
    """
    n = len(train_data["user_id"])
    idx = rng.choice(n, size=n, replace=True)
    return {
        "user_id": [train_data["user_id"][i] for i in idx],
        "question_id": [train_data["question_id"][i] for i in idx],
        "is_correct": [train_data["is_correct"][i] for i in idx],
    }


# ---------------------------------------------------------------------------
# Per-model: train on a bootstrap sample, predict probabilities on `data`
# ---------------------------------------------------------------------------
def knn_predict_probs(boot_matrix, data, k):
    """Train user-based KNNImputer on the bootstrap matrix, return an
    array of predicted probabilities (already continuous in [0, 1],
    since KNNImputer averages 0/1 neighbor values) for every
    (user_id, question_id) pair in `data`.
    """
    imputer = KNNImputer(n_neighbors=k)
    imputed = imputer.fit_transform(boot_matrix)
    probs = np.array([
        imputed[u, q] for u, q in zip(data["user_id"], data["question_id"])
    ])
    return np.clip(probs, 0.0, 1.0)


def irt_predict_probs(boot_dict, valid_data_for_training_curve, data, lr, iterations):
    """Train IRT (theta, beta) on the bootstrap dict, return predicted
    probabilities for every (user_id, question_id) pair in `data`.

    :param valid_data_for_training_curve: irt() expects a validation set
        to print/track progress during training; this does NOT leak into
        the final ensemble prediction, it's only used the same way
        item_response.py's own main() uses it (progress logging).
    """
    theta, beta, _, _, _ = irt(boot_dict, valid_data_for_training_curve, lr, iterations)
    probs = np.array([
        irt_sigmoid(theta[u] - beta[q])
        for u, q in zip(data["user_id"], data["question_id"])
    ])
    return probs


def nn_predict_probs(boot_matrix, num_question, valid_data_for_training_curve, data,
                      k, lr, lamb, num_epoch):
    """Train the AutoEncoder (from neural_network.py, unmodified) on the
    bootstrap matrix, return predicted probabilities for every
    (user_id, question_id) pair in `data`.
    """
    zero_matrix = np.nan_to_num(boot_matrix, nan=0.0)
    train_tensor = torch.FloatTensor(boot_matrix)
    zero_tensor = torch.FloatTensor(zero_matrix)

    model = AutoEncoder(num_question, k)
    nn_train(model, lr, lamb, train_tensor, zero_tensor, valid_data_for_training_curve, num_epoch)

    model.eval()
    probs = np.zeros(len(data["user_id"]))
    for i, u in enumerate(data["user_id"]):
        inputs = zero_tensor[u].unsqueeze(0)
        output = model(inputs)
        probs[i] = output[0][data["question_id"][i]].item()
    return probs


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def accuracy_from_probs(probs, is_correct):
    preds = (np.array(probs) >= 0.5).astype(float)
    return float(np.mean(preds == np.array(is_correct)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_path = "./data"
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    sparse_matrix = load_train_sparse(base_path).toarray()
    num_student, num_question = sparse_matrix.shape

    rng = np.random.default_rng(seed=311)  # fixed seed so results are reproducible

    # ---- Base model 1: kNN, trained on bootstrap sample #1 ----
    boot_mat_knn = bootstrap_matrix(train_data, sparse_matrix.shape, rng)
    knn_valid_probs = knn_predict_probs(boot_mat_knn, valid_data, BEST_KNN_K)
    knn_test_probs = knn_predict_probs(boot_mat_knn, test_data, BEST_KNN_K)
    print(
        f"[kNN alone]  valid_acc={accuracy_from_probs(knn_valid_probs, valid_data['is_correct']):.4f}  "
        f"test_acc={accuracy_from_probs(knn_test_probs, test_data['is_correct']):.4f}"
    )

    # ---- Base model 2: IRT, trained on bootstrap sample #2 ----
    boot_dict_irt = bootstrap_dict(train_data, rng)
    irt_valid_probs = irt_predict_probs(boot_dict_irt, valid_data, valid_data, BEST_IRT_LR, BEST_IRT_ITERATIONS)
    irt_test_probs = irt_predict_probs(boot_dict_irt, valid_data, test_data, BEST_IRT_LR, BEST_IRT_ITERATIONS)
    print(
        f"[IRT alone]  valid_acc={accuracy_from_probs(irt_valid_probs, valid_data['is_correct']):.4f}  "
        f"test_acc={accuracy_from_probs(irt_test_probs, test_data['is_correct']):.4f}"
    )

    # ---- Base model 3: Autoencoder (NN), trained on bootstrap sample #3 ----
    boot_mat_nn = bootstrap_matrix(train_data, sparse_matrix.shape, rng)
    nn_valid_probs = nn_predict_probs(
        boot_mat_nn, num_question, valid_data, valid_data, NN_K, NN_LR, NN_LAMB, NN_NUM_EPOCH
    )
    nn_test_probs = nn_predict_probs(
        boot_mat_nn, num_question, valid_data, test_data, NN_K, NN_LR, NN_LAMB, NN_NUM_EPOCH
    )
    print(
        f"[NN alone]   valid_acc={accuracy_from_probs(nn_valid_probs, valid_data['is_correct']):.4f}  "
        f"test_acc={accuracy_from_probs(nn_test_probs, test_data['is_correct']):.4f}"
    )

    # ---- Ensemble: average the 3 predicted probabilities ----
    ensemble_valid_probs = (knn_valid_probs + irt_valid_probs + nn_valid_probs) / 3.0
    ensemble_test_probs = (knn_test_probs + irt_test_probs + nn_test_probs) / 3.0

    ensemble_valid_acc = accuracy_from_probs(ensemble_valid_probs, valid_data["is_correct"])
    ensemble_test_acc = accuracy_from_probs(ensemble_test_probs, test_data["is_correct"])

    print(f"\n[ENSEMBLE]   valid_acc={ensemble_valid_acc:.4f}  test_acc={ensemble_test_acc:.4f}")


if __name__ == "__main__":
    main()
