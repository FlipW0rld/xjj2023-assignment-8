import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Generate clusters with ellipsoidal shape and shift
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate and shift the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 + distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

# Fit logistic regression and return coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

# Main experiment function
def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}

    n_samples = 8
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        
        # Fit logistic regression and record parameters
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        slope = -beta1 / beta2 if beta2 != 0 else None
        intercept = -beta0 / beta2 if beta2 != 0 else None
        
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Calculate logistic loss
        loss = model.predict_log_proba(X).sum() / len(y)
        loss_list.append(loss)
        
        # Calculate margin width
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices, 
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)
        
        # Plot dataset with decision boundary
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
        
        # Plot decision boundary
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = (-beta0 - beta1 * x_vals) / beta2
        plt.plot(x_vals, y_vals, 'k-', label="Decision Boundary")
        plt.title(f"Shift Distance = {distance}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=24, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=24, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plotting parameters
    plt.figure(figsize=(18, 15))

    # Plot beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list)
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Plot beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list)
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Plot beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list)
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Plot slope
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list)
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Plot intercept
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list)
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list)
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths)
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
