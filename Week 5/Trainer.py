import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


class Trainer:
    def __init__(self, model, metrics):
        self.model = model
        self.config_name = model.__class__.__name__
        self.train_loss_metrics = metrics["training"]["loss"]
        self.train_acc_metrics = metrics["training"]["accuracy"]
        self.test_loss_metrics = metrics["test"]["loss"]
        self.test_acc_metrics = metrics["test"]["accuracy"]

    def training_loop(self, train_ds, test_ds, num_epochs):
        training_losses = []
        training_accuracies = []
        test_losses = []
        test_accuracies = []
        print(f'Starting training for model {self.config_name}')
        for epoch in range(num_epochs):
            # Reset metrics before starting epoch
            self.train_loss_metrics.reset_states()
            self.train_acc_metrics.reset_states()
            self.test_loss_metrics.reset_states()
            self.test_acc_metrics.reset_states()

            # Loop over all batches of training data
            for data in train_ds:
                self.model.train_step(data)

            training_losses.append(self.train_loss_metrics.result())
            training_accuracies.append(self.train_acc_metrics.result())

            # Loop over all batches of test data
            for data in test_ds:
                self.model.test_step(data)

            test_losses.append(self.test_loss_metrics.result())
            test_accuracies.append(self.test_acc_metrics.result())

            print(
                f'- Epoch {epoch + 1} - '
                f'Training Loss: {self.train_loss_metrics.result()}, '
                f'Training Accuracy: {self.train_acc_metrics.result()}, '
                f'Test Loss: {self.test_loss_metrics.result()}, '
                f'Test Accuracy: {self.test_acc_metrics.result()}'
            )

        # Visualize results
        plt.figure()
        line_train_loss, = plt.plot(training_losses)
        line_train_acc, = plt.plot(training_accuracies)
        line_test_loss, = plt.plot(test_losses)
        line_test_acc, = plt.plot(test_accuracies)
        plt.xlabel("Training steps")
        plt.ylabel("Loss / Accuracy")
        plt.legend((line_train_loss, line_train_acc, line_test_loss, line_test_acc),
                   ("Training loss", "Training Accuracy", "Test loss", "Test Accuracy"))
        plt.show()

    def create_summary_writers(self):
        # Define the date-time format for log output
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Folders for the log files
        train_log_path = f"logs/{self.config_name}/{current_time}/train"
        test_log_path = f"logs/{self.config_name}/{current_time}/test"

        # Log writer for training metrics
        train_summary_writer = tf.summary.create_file_writer(train_log_path)

        # Log writer for test metrics
        test_summary_writer = tf.summary.create_file_writer(test_log_path)

        return train_summary_writer, test_summary_writer
