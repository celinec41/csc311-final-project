import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(matrix.T).T
    return sparse_matrix_evaluate(valid_data, imputed_matrix)

def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]

    user_validation_accuracies = []

    print("\nUser-based KNN validation results:")

    for k in k_values:
        val_acc = knn_impute_by_user(
            sparse_matrix,
            val_data,
            k
        )

        user_validation_accuracies.append(val_acc)

        print(
            "k = {}, validation accuracy = {:.4f}".format(
                k,
                val_acc
            )
        )

    # Index of the largest validation accuracy
    best_user_index = np.argmax(user_validation_accuracies)

    # Corresponding best k
    best_user_k = k_values[best_user_index]

    # Evaluate the selected k on the test data
    user_test_accuracy = knn_impute_by_user(
        sparse_matrix,
        test_data,
        best_user_k
    )

    print("\nBest user-based k:", best_user_k)
    print(
        "Best user-based validation accuracy: {:.4f}".format(
            user_validation_accuracies[best_user_index]
        )
    )
    print(
        "User-based test accuracy: {:.4f}".format(
            user_test_accuracy
        )
    )

    item_validation_accuracies = []

    print("\nItem-based KNN validation results:")

    for k in k_values:
        val_acc = knn_impute_by_item(
            sparse_matrix,
            val_data,
            k
        )

        item_validation_accuracies.append(val_acc)

        print(
            "k = {}, validation accuracy = {:.4f}".format(
                k,
                val_acc
            )
        )

    best_item_index = np.argmax(item_validation_accuracies)
    best_item_k = k_values[best_item_index]

    item_test_accuracy = knn_impute_by_item(
        sparse_matrix,
        test_data,
        best_item_k
    )

    print("\nBest item-based k:", best_item_k)
    print(
        "Best item-based validation accuracy: {:.4f}".format(
            item_validation_accuracies[best_item_index]
        )
    )
    print(
        "Item-based test accuracy: {:.4f}".format(
            item_test_accuracy
        )
    )

    # Compare test results

    print("\nFinal comparison:")

    print(
        "User-based test accuracy: {:.4f}".format(
            user_test_accuracy
        )
    )

    print(
        "Item-based test accuracy: {:.4f}".format(
            item_test_accuracy
        )
    )

    if user_test_accuracy > item_test_accuracy:
        print("User-based collaborative filtering performs better.")
    elif item_test_accuracy > user_test_accuracy:
        print("Item-based collaborative filtering performs better.")
    else:
        print("Both methods have the same test accuracy.")

    # Plot validation accuracy

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        user_validation_accuracies,
        marker="o",
        label="User-based KNN"
    )

    plt.plot(
        k_values,
        item_validation_accuracies,
        marker="o",
        label="Item-based KNN"
    )

    plt.xlabel("Number of Neighbours k")
    plt.ylabel("Validation Accuracy")
    plt.title("KNN Validation Accuracy")
    plt.xticks(k_values)
    plt.legend()
    plt.grid()

    plt.savefig("knn_validation_accuracy.png")
    plt.show()


if __name__ == "__main__":
    main()
