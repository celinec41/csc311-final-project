import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # Step 1: Encode the input into a k-dimensional hidden representation.
        # self.g applies the linear transformation W^(1) v + b^(1),
        # then we squash it with sigmoid to get the hidden layer output g(...).
        hidden = torch.sigmoid(self.g(inputs))
 
        # Step 2: Decode the hidden representation back into the original
        # question space (reconstruction of the user's answer vector).
        # self.h applies the linear transformation W^(2) * hidden + b^(2),
        # then sigmoid squashes each output into a probability in [0, 1],
        # representing the predicted probability of answering each question correctly.
        out = torch.sigmoid(self.h(hidden))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

def compute_valid_objective(model, zero_train_data, valid_data):
    model.eval()
    total_loss = 0.0
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(zero_train_data[u]).unsqueeze(0)
        output = model(inputs)
        q = valid_data["question_id"][i]
        c = valid_data["is_correct"][i]
        total_loss += (output[0][q].item() - c) ** 2.0
    return total_loss

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    train_obj_history = []
    valid_obj_history = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_obj = compute_valid_objective(model, zero_train_data, valid_data)
        train_obj_history.append(train_loss)
        valid_obj_history.append(valid_obj)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
    return train_obj_history, valid_obj_history
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)



def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_question = train_matrix.shape[1]
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    '''
    # Set model hyperparameters.
    lr_list = [0.001, 0.003, 0.01, 0.03, 0.1]
    num_epoch = 50

    best_valid_acc = -1.0
    best_lr, best_k = None, None

    for lr in lr_list:
        for k in range(15, 100, 5):
            model = AutoEncoder(num_question, k)
            train(model, lr, 0.0, train_matrix, zero_train_matrix, valid_data, num_epoch)
            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            print(f"lr={lr}, k={k}: valid_acc={valid_acc:.4f}")
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_lr, best_k = lr, k

    print(f"Best (lr*, k*) = ({best_lr}, {best_k}), valid_acc={best_valid_acc:.4f}")
 
    '''   
    # Set model hyperparameters.
    best_k = 30
    final_model = AutoEncoder(num_question, best_k)

    # Set optimization hyperparameters.
    best_lr = 0.01
    num_epoch = 50
    lamb = None

    train_obj_hist, valid_obj_hist = train(
        final_model, best_lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch
    )

    test_acc = evaluate(final_model, zero_train_matrix, test_data)
    print(f"Final test accuracy (k*={best_k}, lr*={best_lr}): {test_acc:.4f}")

    epochs = list(range(1, num_epoch + 1))
    plt.figure()
    plt.plot(epochs, train_obj_hist, label="Training objective")
    plt.plot(epochs, valid_obj_hist, label="Validation objective")
    plt.xlabel("Epoch")
    plt.ylabel("Squared-error objective")
    plt.legend()
    plt.savefig("nn_objective_curve.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
