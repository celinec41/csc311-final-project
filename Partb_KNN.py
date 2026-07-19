"""Part B: validation-selected improvements to the Part A KNN models.

The experiment compares distance-weighted KNN with a hybrid that combines the
Part A user- and item-based predictions and a question-correctness prior.  All
hyperparameters are selected using validation accuracy.  The public test set is
not loaded until the winning validation configuration has been fixed.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

from utils import (
    evaluate,
    load_public_test_csv,
    load_train_sparse,
    load_valid_csv,
)


DISTANCE_K_VALUES = (5, 10, 15, 20, 25, 30, 40, 50)
USER_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
KNN_WEIGHTS = (0.7, 0.8, 0.9, 1.0)
THRESHOLDS = (0.45, 0.475, 0.5, 0.525, 0.55)


@dataclass(frozen=True)
class KNNConfig:
    """Configuration of one user- or item-based KNN imputer."""

    orientation: str
    k: int
    weights: str = "uniform"


@dataclass(frozen=True)
class HybridConfig:
    """Hyperparameters of the final hybrid predictor."""

    user_config: KNNConfig
    item_config: KNNConfig
    user_weight: float
    knn_weight: float
    threshold: float


PART_A_USER = KNNConfig("user", k=11)
PART_A_ITEM = KNNConfig("item", k=21)


def extract_probabilities(data, matrix):
    """Return matrix predictions for the pairs listed in ``data``."""
    users = np.asarray(data["user_id"], dtype=int)
    questions = np.asarray(data["question_id"], dtype=int)
    return np.asarray(matrix[users, questions], dtype=float)


def impute_response_matrix(matrix, config):
    """Impute the response matrix using a user- or item-based KNN."""
    if config.orientation not in {"user", "item"}:
        raise ValueError("orientation must be either 'user' or 'item'")

    imputer = KNNImputer(
        n_neighbors=config.k,
        weights=config.weights,
        metric="nan_euclidean",
    )

    if config.orientation == "user":
        return imputer.fit_transform(matrix)
    return imputer.fit_transform(matrix.T).T


def compute_question_priors(matrix):
    """Estimate every question's correctness probability from training data."""
    observed = ~np.isnan(matrix)
    counts = observed.sum(axis=0)
    correct_counts = np.nansum(matrix, axis=0)
    global_prior = float(correct_counts.sum() / counts.sum())

    return np.divide(
        correct_counts,
        counts,
        out=np.full(matrix.shape[1], global_prior, dtype=float),
        where=counts > 0,
    )


def accuracy(labels, probabilities, threshold=0.5):
    """Compute binary accuracy for a probability vector."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(probabilities) >= threshold
    return float(np.mean(predictions == labels))


def candidate_configs():
    """Return the Part A baselines and Part B distance-weighted candidates."""
    weighted = [
        KNNConfig(orientation, k, weights="distance")
        for orientation in ("user", "item")
        for k in DISTANCE_K_VALUES
    ]
    return [PART_A_USER, PART_A_ITEM, *weighted]


def fit_predictions(matrix, data, configs, stage):
    """Fit the requested imputers and return their predictions on ``data``."""
    predictions = {}
    configs = list(configs)

    for index, config in enumerate(configs, start=1):
        print(
            "[{} {}/{}] {} KNN: k={}, weights={}".format(
                stage,
                index,
                len(configs),
                config.orientation,
                config.k,
                config.weights,
            )
        )
        matrix_hat = impute_response_matrix(matrix, config)
        predictions[config] = extract_probabilities(data, matrix_hat)

    return predictions


def hybrid_probabilities(user_prob, item_prob, prior_prob, config):
    """Combine user KNN, item KNN, and question-prior probabilities."""
    knn_prob = (
        config.user_weight * user_prob
        + (1.0 - config.user_weight) * item_prob
    )
    return config.knn_weight * knn_prob + (1.0 - config.knn_weight) * prior_prob


def select_hybrid(validation_predictions, valid_data, valid_priors):
    """Select the hybrid configuration using validation labels only."""
    labels = valid_data["is_correct"]
    user_configs = [
        config
        for config in validation_predictions
        if config.orientation == "user"
    ]
    item_configs = [
        config
        for config in validation_predictions
        if config.orientation == "item"
    ]

    best_config = None
    best_accuracy = -1.0

    for user_config in user_configs:
        user_prob = validation_predictions[user_config]
        for item_config in item_configs:
            item_prob = validation_predictions[item_config]
            for user_weight in USER_WEIGHTS:
                for knn_weight in KNN_WEIGHTS:
                    for threshold in THRESHOLDS:
                        config = HybridConfig(
                            user_config=user_config,
                            item_config=item_config,
                            user_weight=user_weight,
                            knn_weight=knn_weight,
                            threshold=threshold,
                        )
                        probabilities = hybrid_probabilities(
                            user_prob, item_prob, valid_priors, config
                        )
                        score = accuracy(labels, probabilities, threshold)
                        if score > best_accuracy:
                            best_config = config
                            best_accuracy = score

    return best_config, best_accuracy


def best_weighted_config(validation_predictions, valid_data, orientation):
    """Return the best distance-weighted KNN of one orientation."""
    labels = valid_data["is_correct"]
    configs = [
        config
        for config in validation_predictions
        if config.orientation == orientation and config.weights == "distance"
    ]
    return max(
        configs,
        key=lambda config: accuracy(labels, validation_predictions[config]),
    )


def unique_configs(configs):
    """Remove duplicate configurations while preserving their order."""
    return list(dict.fromkeys(configs))


def save_comparison_plot(names, valid_scores, test_scores):
    """Save the Part A/Part B accuracy comparison used in the report."""
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    valid_bars = ax.bar(x - width / 2, valid_scores, width, label="Validation")
    test_bars = ax.bar(x + width / 2, test_scores, width, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_title("Part A KNN Baselines vs. Part B Improved KNN")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0.60, max(valid_scores + test_scores) + 0.025)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(valid_bars, fmt="%.4f", padding=3, fontsize=8)
    ax.bar_label(test_bars, fmt="%.4f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig("partb_knn_comparison.png", dpi=180)
    plt.close(fig)


def save_tuning_plot(valid_data, user_prob, item_prob, prior_prob):
    """Visualize validation selection of alpha, gamma, and the threshold."""
    labels = valid_data["is_correct"]
    best_scores_by_alpha = []

    for user_weight in USER_WEIGHTS:
        hybrid = user_weight * user_prob + (1.0 - user_weight) * item_prob
        scores = [
            accuracy(
                labels,
                knn_weight * hybrid + (1.0 - knn_weight) * prior_prob,
                threshold,
            )
            for knn_weight in KNN_WEIGHTS
            for threshold in THRESHOLDS
        ]
        best_scores_by_alpha.append(max(scores))

    selected_alpha = USER_WEIGHTS[int(np.argmax(best_scores_by_alpha))]
    selected_hybrid = (
        selected_alpha * user_prob + (1.0 - selected_alpha) * item_prob
    )
    heatmap = np.asarray([
        [
            accuracy(
                labels,
                knn_weight * selected_hybrid
                + (1.0 - knn_weight) * prior_prob,
                threshold,
            )
            for threshold in THRESHOLDS
        ]
        for knn_weight in KNN_WEIGHTS
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(USER_WEIGHTS, best_scores_by_alpha, marker="o")
    axes[0].axvline(selected_alpha, color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel(r"User-KNN weight $\alpha$")
    axes[0].set_ylabel("Best validation accuracy")
    axes[0].set_title(r"Selecting $\alpha$")
    axes[0].set_xticks(USER_WEIGHTS)
    axes[0].grid(alpha=0.3)

    image = axes[1].imshow(heatmap, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(THRESHOLDS)), THRESHOLDS)
    axes[1].set_yticks(range(len(KNN_WEIGHTS)), KNN_WEIGHTS)
    axes[1].set_xlabel(r"Classification threshold $\tau$")
    axes[1].set_ylabel(r"KNN weight $\gamma$")
    axes[1].set_title(
        r"Validation accuracy at $\alpha={}$".format(selected_alpha)
    )
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            axes[1].text(
                column,
                row,
                "{:.4f}".format(heatmap[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if heatmap[row, column] > heatmap.mean() else "black",
            )
    fig.colorbar(image, ax=axes[1], label="Validation accuracy")
    fig.tight_layout()
    fig.savefig("partb_knn_tuning.png", dpi=180)
    plt.close(fig)


def print_results(names, valid_scores, test_scores, best_config):
    """Print the selected configuration and a report-ready result table."""
    print("\nSelected Part B configuration")
    print("  user KNN:   k={}, weights={}".format(
        best_config.user_config.k, best_config.user_config.weights
    ))
    print("  item KNN:   k={}, weights={}".format(
        best_config.item_config.k, best_config.item_config.weights
    ))
    print("  user weight (alpha): {:.3f}".format(best_config.user_weight))
    print("  KNN weight (gamma):  {:.3f}".format(best_config.knn_weight))
    print("  threshold (tau):     {:.3f}".format(best_config.threshold))

    print("\n{:<24s} {:>12s} {:>12s}".format(
        "Model", "Validation", "Test"
    ))
    print("-" * 50)
    for name, valid_score, test_score in zip(names, valid_scores, test_scores):
        print("{:<24s} {:>12.4f} {:>12.4f}".format(
            name, valid_score, test_score
        ))


def main():
    matrix = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    priors = compute_question_priors(matrix)
    valid_priors = priors[np.asarray(valid_data["question_id"], dtype=int)]

    # Phase 1: fit/tune using training and validation data only.
    validation_predictions = fit_predictions(
        matrix,
        valid_data,
        candidate_configs(),
        stage="validation",
    )
    best_config, best_valid_score = select_hybrid(
        validation_predictions, valid_data, valid_priors
    )
    best_weighted_user = best_weighted_config(
        validation_predictions, valid_data, "user"
    )
    best_weighted_item = best_weighted_config(
        validation_predictions, valid_data, "item"
    )

    # Phase 2: the test set is loaded only after validation selection is final.
    test_data = load_public_test_csv("./data")
    test_priors = priors[np.asarray(test_data["question_id"], dtype=int)]
    required_test_configs = unique_configs([
        PART_A_USER,
        PART_A_ITEM,
        best_weighted_user,
        best_weighted_item,
        best_config.user_config,
        best_config.item_config,
    ])
    test_predictions = fit_predictions(
        matrix,
        test_data,
        required_test_configs,
        stage="test",
    )

    valid_hybrid = 0.5 * (
        validation_predictions[PART_A_USER]
        + validation_predictions[PART_A_ITEM]
    )
    test_hybrid = 0.5 * (
        test_predictions[PART_A_USER] + test_predictions[PART_A_ITEM]
    )
    valid_smoothed = 0.9 * valid_hybrid + 0.1 * valid_priors
    test_smoothed = 0.9 * test_hybrid + 0.1 * test_priors

    final_valid_prob = hybrid_probabilities(
        validation_predictions[best_config.user_config],
        validation_predictions[best_config.item_config],
        valid_priors,
        best_config,
    )
    final_test_prob = hybrid_probabilities(
        test_predictions[best_config.user_config],
        test_predictions[best_config.item_config],
        test_priors,
        best_config,
    )

    names = [
        "Part A user",
        "Part A item",
        "Weighted user",
        "Weighted item",
        "50-50 hybrid",
        "Hybrid + prior",
        "Final hybrid",
    ]
    valid_scores = [
        evaluate(valid_data, validation_predictions[PART_A_USER]),
        evaluate(valid_data, validation_predictions[PART_A_ITEM]),
        evaluate(valid_data, validation_predictions[best_weighted_user]),
        evaluate(valid_data, validation_predictions[best_weighted_item]),
        evaluate(valid_data, valid_hybrid),
        evaluate(valid_data, valid_smoothed),
        evaluate(valid_data, final_valid_prob, best_config.threshold),
    ]
    test_scores = [
        evaluate(test_data, test_predictions[PART_A_USER]),
        evaluate(test_data, test_predictions[PART_A_ITEM]),
        evaluate(test_data, test_predictions[best_weighted_user]),
        evaluate(test_data, test_predictions[best_weighted_item]),
        evaluate(test_data, test_hybrid),
        evaluate(test_data, test_smoothed),
        evaluate(test_data, final_test_prob, best_config.threshold),
    ]

    # Guard against an accidental mismatch between selection and reporting.
    if not np.isclose(valid_scores[-1], best_valid_score):
        raise RuntimeError("Reported validation score differs from selected score")

    print("\nBest distance-weighted user:", best_weighted_user)
    print("Best distance-weighted item:", best_weighted_item)
    print_results(names, valid_scores, test_scores, best_config)
    save_tuning_plot(
        valid_data,
        validation_predictions[PART_A_USER],
        validation_predictions[PART_A_ITEM],
        valid_priors,
    )
    save_comparison_plot(names, valid_scores, test_scores)


if __name__ == "__main__":
    main()
