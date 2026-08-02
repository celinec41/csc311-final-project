from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

from utils import (
    evaluate,
    load_public_test_csv,
    load_train_sparse,
    load_valid_csv,
)


# k values tried for distance-weighted KNN, in addition to the fixed Part A
# uniform-weighted baselines (k=11 for user, k=21 for item).
DISTANCE_K_VALUES = (5, 10, 15, 20, 25, 30, 40, 50)

USER_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # alpha: user-KNN vs. item-KNN
KNN_WEIGHTS = (0.7, 0.8, 0.9, 1.0)  # gamma: KNN mix vs. prior mix
PRIOR_BETA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)  # beta: question vs. student prior
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
    user_weight: float  # alpha
    knn_weight: float  # gamma
    prior_beta: float  # beta
    threshold: float  # tau


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


def compute_student_priors(matrix):
    """Estimate every student's correctness probability from training data."""
    observed = ~np.isnan(matrix)
    counts = observed.sum(axis=1)
    correct_counts = np.nansum(matrix, axis=1)
    global_prior = float(correct_counts.sum() / counts.sum())

    return np.divide(
        correct_counts,
        counts,
        out=np.full(matrix.shape[0], global_prior, dtype=float),
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


def hybrid_probabilities(user_prob, item_prob, question_prior, student_prior, config):
    """Combine user KNN, item KNN, and the two correctness priors."""
    knn_prob = (
        config.user_weight * user_prob + (1.0 - config.user_weight) * item_prob
    )
    combined_prior = (
        config.prior_beta * question_prior
        + (1.0 - config.prior_beta) * student_prior
    )
    return config.knn_weight * knn_prob + (1.0 - config.knn_weight) * combined_prior


def select_hybrid(
    validation_predictions, valid_data, valid_question_prior, valid_student_prior
):
    """Select the hybrid configuration using validation labels only."""
    labels = np.asarray(valid_data["is_correct"], dtype=int)
    user_configs = [
        c for c in validation_predictions if c.orientation == "user"
    ]
    item_configs = [
        c for c in validation_predictions if c.orientation == "item"
    ]

    # Precompute the beta-blended prior once per beta value; it does not
    # depend on the KNN configuration or on alpha.
    combined_priors = {
        beta: beta * valid_question_prior + (1.0 - beta) * valid_student_prior
        for beta in PRIOR_BETA_VALUES
    }

    best_config = None
    best_accuracy = -1.0

    for user_config in user_configs:
        user_prob = validation_predictions[user_config]
        for item_config in item_configs:
            item_prob = validation_predictions[item_config]
            for user_weight in USER_WEIGHTS:
                knn_prob = user_weight * user_prob + (1.0 - user_weight) * item_prob
                for knn_weight in KNN_WEIGHTS:
                    for beta, prior in combined_priors.items():
                        probabilities = (
                            knn_weight * knn_prob + (1.0 - knn_weight) * prior
                        )
                        for threshold in THRESHOLDS:
                            score = accuracy(labels, probabilities, threshold)
                            if score > best_accuracy:
                                best_config = HybridConfig(
                                    user_config=user_config,
                                    item_config=item_config,
                                    user_weight=user_weight,
                                    knn_weight=knn_weight,
                                    prior_beta=beta,
                                    threshold=threshold,
                                )
                                best_accuracy = score

    return best_config, best_accuracy


def best_solo_config(validation_predictions, valid_data, orientation):
    """Return the best single KNN configuration of one orientation
    (uniform or distance, any k), selected on validation accuracy."""
    labels = valid_data["is_correct"]
    configs = [
        c for c in validation_predictions if c.orientation == orientation
    ]
    return max(
        configs,
        key=lambda c: accuracy(labels, validation_predictions[c]),
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


def save_tuning_plot(
    valid_data, best_config, user_prob, item_prob, question_prior, student_prior
):
    """Visualize validation sensitivity to beta and to (gamma, threshold),
    holding the other hyperparameters at their selected values."""
    labels = valid_data["is_correct"]
    knn_prob = (
        best_config.user_weight * user_prob
        + (1.0 - best_config.user_weight) * item_prob
    )

    beta_scores = [
        accuracy(
            labels,
            best_config.knn_weight
            * knn_prob
            + (1.0 - best_config.knn_weight)
            * (beta * question_prior + (1.0 - beta) * student_prior),
            best_config.threshold,
        )
        for beta in PRIOR_BETA_VALUES
    ]

    combined_prior = (
        best_config.prior_beta * question_prior
        + (1.0 - best_config.prior_beta) * student_prior
    )
    heatmap = np.asarray(
        [
            [
                accuracy(
                    labels,
                    knn_weight * knn_prob + (1.0 - knn_weight) * combined_prior,
                    threshold,
                )
                for threshold in THRESHOLDS
            ]
            for knn_weight in KNN_WEIGHTS
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(PRIOR_BETA_VALUES, beta_scores, marker="o")
    axes[0].axvline(best_config.prior_beta, color="tab:red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel(r"Question-prior weight $\beta$ (1$-\beta$ = student prior)")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title(r"Selecting $\beta$")
    axes[0].set_xticks(PRIOR_BETA_VALUES)
    axes[0].grid(alpha=0.3)

    image = axes[1].imshow(heatmap, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(THRESHOLDS)), THRESHOLDS)
    axes[1].set_yticks(range(len(KNN_WEIGHTS)), KNN_WEIGHTS)
    axes[1].set_xlabel(r"Classification threshold $\tau$")
    axes[1].set_ylabel(r"KNN weight $\gamma$")
    axes[1].set_title(
        r"Validation accuracy at $\alpha={:.2f}$, $\beta={:.2f}$".format(
            best_config.user_weight, best_config.prior_beta
        )
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
    print("  user weight (alpha):        {:.3f}".format(best_config.user_weight))
    print("  KNN weight (gamma):         {:.3f}".format(best_config.knn_weight))
    print("  question-prior weight (beta): {:.3f}".format(best_config.prior_beta))
    print("  threshold (tau):            {:.3f}".format(best_config.threshold))

    print("\n{:<28s} {:>12s} {:>12s}".format("Model", "Validation", "Test"))
    print("-" * 54)
    for name, valid_score, test_score in zip(names, valid_scores, test_scores):
        print("{:<28s} {:>12.4f} {:>12.4f}".format(name, valid_score, test_score))


def main():
    matrix = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    question_priors = compute_question_priors(matrix)
    student_priors = compute_student_priors(matrix)
    valid_question_prior = question_priors[
        np.asarray(valid_data["question_id"], dtype=int)
    ]
    valid_student_prior = student_priors[
        np.asarray(valid_data["user_id"], dtype=int)
    ]

    # Phase 1: fit/tune using training and validation data only.
    validation_predictions = fit_predictions(
        matrix,
        valid_data,
        candidate_configs(),
        stage="validation",
    )
    best_config, best_valid_score = select_hybrid(
        validation_predictions, valid_data, valid_question_prior, valid_student_prior
    )
    best_solo_user = best_solo_config(validation_predictions, valid_data, "user")
    best_solo_item = best_solo_config(validation_predictions, valid_data, "item")

    # Phase 2: the test set is loaded only after validation selection is final.
    test_data = load_public_test_csv("./data")
    test_question_prior = question_priors[
        np.asarray(test_data["question_id"], dtype=int)
    ]
    test_student_prior = student_priors[np.asarray(test_data["user_id"], dtype=int)]
    required_test_configs = unique_configs([
        PART_A_USER,
        PART_A_ITEM,
        best_solo_user,
        best_solo_item,
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
        validation_predictions[PART_A_USER] + validation_predictions[PART_A_ITEM]
    )
    test_hybrid = 0.5 * (
        test_predictions[PART_A_USER] + test_predictions[PART_A_ITEM]
    )

    question_only_config = replace(best_config, prior_beta=1.0)
    student_only_config = replace(best_config, prior_beta=0.0)

    valid_question_only = hybrid_probabilities(
        validation_predictions[best_config.user_config],
        validation_predictions[best_config.item_config],
        valid_question_prior,
        valid_student_prior,
        question_only_config,
    )
    test_question_only = hybrid_probabilities(
        test_predictions[best_config.user_config],
        test_predictions[best_config.item_config],
        test_question_prior,
        test_student_prior,
        question_only_config,
    )
    valid_student_only = hybrid_probabilities(
        validation_predictions[best_config.user_config],
        validation_predictions[best_config.item_config],
        valid_question_prior,
        valid_student_prior,
        student_only_config,
    )
    test_student_only = hybrid_probabilities(
        test_predictions[best_config.user_config],
        test_predictions[best_config.item_config],
        test_question_prior,
        test_student_prior,
        student_only_config,
    )

    final_valid_prob = hybrid_probabilities(
        validation_predictions[best_config.user_config],
        validation_predictions[best_config.item_config],
        valid_question_prior,
        valid_student_prior,
        best_config,
    )
    final_test_prob = hybrid_probabilities(
        test_predictions[best_config.user_config],
        test_predictions[best_config.item_config],
        test_question_prior,
        test_student_prior,
        best_config,
    )

    names = [
        "Part A user",
        "Part A item",
        "Best tuned user",
        "Best tuned item",
        "50-50 hybrid",
        "Hybrid + question prior",
        "Hybrid + student prior",
        "Final hybrid (both priors)",
    ]
    valid_scores = [
        evaluate(valid_data, validation_predictions[PART_A_USER]),
        evaluate(valid_data, validation_predictions[PART_A_ITEM]),
        evaluate(valid_data, validation_predictions[best_solo_user]),
        evaluate(valid_data, validation_predictions[best_solo_item]),
        evaluate(valid_data, valid_hybrid),
        evaluate(valid_data, valid_question_only, best_config.threshold),
        evaluate(valid_data, valid_student_only, best_config.threshold),
        evaluate(valid_data, final_valid_prob, best_config.threshold),
    ]
    test_scores = [
        evaluate(test_data, test_predictions[PART_A_USER]),
        evaluate(test_data, test_predictions[PART_A_ITEM]),
        evaluate(test_data, test_predictions[best_solo_user]),
        evaluate(test_data, test_predictions[best_solo_item]),
        evaluate(test_data, test_hybrid),
        evaluate(test_data, test_question_only, best_config.threshold),
        evaluate(test_data, test_student_only, best_config.threshold),
        evaluate(test_data, final_test_prob, best_config.threshold),
    ]

    # Guard against an accidental mismatch between selection and reporting.
    if not np.isclose(valid_scores[-1], best_valid_score):
        raise RuntimeError("Reported validation score differs from selected score")

    print("\nBest solo user config:", best_solo_user)
    print("Best solo item config:", best_solo_item)
    print_results(names, valid_scores, test_scores, best_config)
    save_tuning_plot(
        valid_data,
        best_config,
        validation_predictions[best_config.user_config],
        validation_predictions[best_config.item_config],
        valid_question_prior,
        valid_student_prior,
    )
    save_comparison_plot(names, valid_scores, test_scores)


if __name__ == "__main__":
    main()
