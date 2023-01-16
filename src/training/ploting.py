import matplotlib.pyplot as plt

class Ploting:
    @staticmethod
    def plot_loss(train_losses: list, test_losses: list) -> None:
        """Plot the loss.

        Args:
            train_losses (list): Train loss.
            test_losses (list): Test loss.
        """
        
        plt.plot(train_losses, label="Train loss")
        plt.plot(test_losses, label="Test loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(train_accuracies: list, test_accuracies: list) -> None:
        """Plot the accuracy.

        Args:
            train_accuracies (list): Train accuracy.
            test_accuracies (list): Test accuracy.
        """

        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_loss_accuracy(train_losses: list, train_accuracies: list,
                           test_losses: list, test_accuracies: list) -> None:
        """Plot the loss and the accuracy.
        
        Args:
            train_losses (list): Train loss.
            train_accuracies (list): Train accuracy.
            test_losses (list): Test loss.
            test_accuracies (list): Test accuracy.
        """

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(train_losses, color=color, label="Train loss")
        ax1.plot(test_losses, color=color, linestyle='dashed', label="Test loss")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc="upper left")

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(train_accuracies, color=color, label="Train accuracy")
        ax2.plot(test_accuracies, color=color, linestyle='dashed', label="Test accuracy")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc="upper right")
        
        fig.tight_layout()
        plt.show()