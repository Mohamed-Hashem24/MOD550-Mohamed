from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

karg = {'observed': observed, 'predicted': predicted}

factory = {'mse_vanilla' : vanilla_mse,
           'mse_numpy' : numpy_mse,
           'mse_sk' : lambda observed, predicted: sk_mse(y_true=observed, y_pred=predicted)
           }

for talker, worker in factory.items():
    exec_time = it.timeit(lambda: worker(**karg), number=100) / 100
    mse = worker(**karg)
    print(f"Mean Squared Error, {talker} :", mse, 
          f"Average execution time: {exec_time} seconds")


# Step 1: Generate 1D oscillatory data with and without noise
def generate_oscillatory_data(num_points=1000, noise_level=0.2):
    x = np.linspace(0, 10, num_points)
    y_clean = np.sin(x)  # Clean oscillatory function (e.g., sine wave)
    y_noisy = y_clean + noise_level * np.random.randn(num_points)  # Add Gaussian noise
    return x, y_clean, y_noisy

# Step 2: Linear Regression (LR)
def linear_regression(x, y):
    x = x.reshape(-1, 1)  # Reshape for sklearn
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, vanilla_mse(y, y_pred)  # Return predictions and loss

# Step 3: Neural Network (NN)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_neural_network(x, y, epochs=1000, learning_rate=0.01):
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    x, y_clean, y_noisy = generate_oscillatory_data(num_points=1000, noise_level=0.2)

    # Store predictions, errors, and losses during training
    predictions = []
    errors = []
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        # Store predictions, errors, and losses every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                y_pred_np = model(x_tensor).numpy()
                predictions.append(y_pred_np)
                error = vanilla_mse(y_clean, y_pred_np)
                errors.append(error)
                losses.append(loss.item())

    return predictions, errors, losses

# Step 4: Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_pinn(x, y, epochs=1000, learning_rate=0.01):
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    model = PINN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    x, y_clean, y_noisy = generate_oscillatory_data(num_points=1000, noise_level=0.2)

    # Store predictions, errors, and losses during training
    predictions = []
    errors = []
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_tensor)

        # Data loss
        data_loss = criterion(y_pred, y_tensor)

        # Physics loss (enforce the sine wave equation: d^2y/dx^2 + y = 0)
        dy_dx = torch.autograd.grad(y_pred, x_tensor, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x_tensor, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
        physics_loss = torch.mean((d2y_dx2 + y_pred) ** 2)

        # Total loss
        total_loss = data_loss + physics_loss
        total_loss.backward()
        optimizer.step()

        # Store predictions, errors, and losses every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                y_pred_np = model(x_tensor).numpy()
                predictions.append(y_pred_np)
                error = vanilla_mse(y_clean, y_pred_np)
                errors.append(error)
                losses.append(total_loss.item())

    return predictions, errors, losses

# Step 5: Clustering
def perform_clustering(x, y_noisy, max_clusters=10):
    data = np.column_stack((x, y_noisy))  # Combine x and y_noisy for clustering
    variances = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        variances.append(kmeans.inertia_)  # Inertia is the within-cluster sum of squares

    # Plot variance vs number of clusters
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), variances, marker="o")
    plt.title("Variance vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Variance (Inertia)")
    plt.grid()
    plt.show()

    return variances

# Step 6: Plot regression function as a function of iteration numbers
def plot_regression_progress(x, y_clean, predictions, method_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_clean, label="Clean Data", color="blue")
    for i, pred in enumerate(predictions):
        plt.plot(x, pred, label=f"{method_name} Iter {i * 10}", alpha=0.5)
    plt.title(f"{method_name} Predictions Over Iterations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Step 7: Plot error and loss as a function of iteration number
def plot_error_and_loss(errors, losses, method_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, 1000, 10), errors, label=f"{method_name} Error", marker="o")
    plt.plot(range(0, 1000, 10), losses, label=f"{method_name} Loss", marker="o")
    plt.title(f"{method_name} Error and Loss vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# Step 8: Plot number of iterations needed to converge as a function of learning rate
def plot_convergence_vs_learning_rate(x, y_noisy, learning_rates):
    iterations_to_converge = []
    for lr in learning_rates:
        _, _, losses = train_neural_network(x, y_noisy, epochs=1000, learning_rate=lr)
        # Find the iteration where loss stabilizes (e.g., changes by less than 1e-5)
        for i in range(1, len(losses)):
            if abs(losses[i] - losses[i - 1]) < 1e-5:
                iterations_to_converge.append(i * 10)
                break
        else:
            iterations_to_converge.append(1000)  # Default to max epochs if no convergence

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, iterations_to_converge, marker="o")
    plt.title("Iterations to Converge vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations to Converge")
    plt.grid()
    plt.show()

# Main function to execute the steps
def main():
    # Step 1: Generate data
    x, y_clean, y_noisy = generate_oscillatory_data(num_points=1000, noise_level=0.2)
    print("Test successful")  # Task 1
    print(f"Data generated: n_points=1000, range=[0, 10], noise_level=0.2")  # Task 2

    # Step 2: Perform clustering
    variances = perform_clustering(x, y_noisy)
    print("Clustering method: KMeans, Parameters: n_clusters=1 to 10, random_state=42")  # Task 3

    # Step 3: Linear Regression
    y_pred_lr, lr_loss = linear_regression(x, y_noisy)
    print("Task completed: Linear Regression")  # Task 4

    # Step 4: Neural Network
    nn_predictions, nn_errors, nn_losses = train_neural_network(x, y_noisy)
    print("Task completed: Neural Network")  # Task 4

    # Step 5: PINN
    pinn_predictions, pinn_errors, pinn_losses = train_pinn(x, y_noisy)
    print("Task completed: PINN")  # Task 4

    # Step 6: Plot regression progress
    plot_regression_progress(x, y_clean, nn_predictions, "Neural Network")  # Task 5
    plot_regression_progress(x, y_clean, pinn_predictions, "PINN")  # Task 5

    # Step 7: Plot error and loss
    plot_error_and_loss(nn_errors, nn_losses, "Neural Network")  # Task 6 and 7
    plot_error_and_loss(pinn_errors, pinn_losses, "PINN")  # Task 6 and 7

    # Step 8: Plot convergence vs learning rate
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    plot_convergence_vs_learning_rate(x, y_noisy, learning_rates)  # Task 8

if __name__ == "__main__":
    main()