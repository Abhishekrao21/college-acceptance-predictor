# college-acceptance-predictor
Developed a robust logistic regression model using Python and various data science libraries to predict student admission based on their exam scores. This model demonstrates proficiency in machine learning algorithms, data visualization, and optimization techniques. 

Logistic Regression Model for Admission Prediction
Project Overview:

Developed a robust logistic regression model using Python and various data science libraries to predict student admission based on their exam scores. This model demonstrates proficiency in machine learning algorithms, data visualization, and optimization techniques. Below is a detailed breakdown of the project's workflow and achievements:

Key Features and Accomplishments:

Data Visualization:

Utilized matplotlib to plot the dataset, effectively differentiating between admitted and non-admitted students. This visualization helps in understanding the data distribution and identifying potential decision boundaries.
Implementation of Logistic Regression:

Defined the logistic sigmoid function to transform the input data into probabilities.
Computed the cost function using cross-entropy loss, ensuring the model's predictions are well-calibrated probabilities.
Gradient Descent Optimization:

Implemented gradient descent from scratch to iteratively update the model's parameters (weights and bias). This ensures the model converges to the optimal solution.
Tracked the cost function's value across iterations to monitor convergence and adjust learning parameters if necessary.
Decision Boundary Visualization:

Plotted the decision boundary over the data points, providing a clear visual representation of the model's classification capabilities.
Model Performance:

Achieved an impressive training accuracy, indicating the model's effectiveness in predicting admissions based on the given features.
Technical Breakdown:

Data Handling:

python
X_train = np.array([...])
y_train = np.array([...])
Loaded and structured the dataset for training, ensuring it is ready for model consumption.
Plotting Data Points:

python
plot_data(X_train, y_train, pos_label="Admitted", neg_label="Not admitted")
Visualized the exam scores of admitted and non-admitted students, aiding in exploratory data analysis.
Sigmoid Function:

python
def sigmoid(z):
    ...
Implemented the sigmoid function to map input features to a probability range between 0 and 1.
Cost Function Calculation:

python
def compute_cost(X, y, w, b):
    ...
Calculated the cost to quantify the difference between predicted and actual values, guiding the optimization process.
Gradient Computation:

python
def compute_gradient(X, y, w, b):
    ...
Derived gradients for the weights and bias, crucial for the gradient descent algorithm to update parameters efficiently.
Gradient Descent Implementation:

python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    ...
Iteratively optimized the model parameters to minimize the cost function, ensuring high model accuracy.
Decision Boundary Plotting:

python

plot_decision_boundary(w, b, X_train, y_train)
Visualized the model's decision boundary, showcasing its classification capability.
Model Prediction and Accuracy:

python
def predict(X, w, b):
    ...
p = predict(X_train, w, b)
print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))
Predicted admission probabilities and calculated the model's accuracy, validating its performance on the training data.
