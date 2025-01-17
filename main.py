import numpy as np
import matplotlib.pyplot as plt
from utils import *
import math
def plot_data(X, y, pos_label, neg_label):
    # Find indices of positive and negative examples
    pos = y == 1
    neg = y == 0

    # Plot positive examples
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label=pos_label)
    # Plot negative examples
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label=neg_label)
def plot_decision_boundary(w, b, X, y):
    # Plot the original data
    plot_data(X, y, pos_label="Admitted", neg_label="Not admitted")

    # Create a grid of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calculate the predictions for each point in the grid
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='g')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def sigmoid(z):
    e=2.718281828459045
    a=-z
    a=e**a
    a=a+1
    a=1/a
    g=a
    return g
def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    j = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        a = 1 / (1 + np.exp(-z))
        l = (-y[i] * np.log(a) - (1 - y[i]) * np.log(1 - a))
        j += l
    
    total_cost = j / m
    
    return total_cost
def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z = np.dot(X[i],w) + b
        a = 1/(1+np.exp(-z))
        dj_db+=a-y[i]
        for j in range(n):
            dj_dw[j]+=(a-y[i])*X[i,j]
    dj_dw /= m
    dj_db /= m  
    return dj_db, dj_dw
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 

def predict(X, w, b): 
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += w[j]*X[i,j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = 1/(1+np.exp(-z_wb))

        # Apply the threshold
        p[i] = 1 if f_wb>=0.5 else 0

    return p



X_train = np.array([[34.62365962, 78.02469282],[30.28671077, 43.89499752],[35.84740877, 72.90219803],[60.18259939, 86.3085521 ],[79.03273605, 75.34437644],[45.08327748, 56.31637178],[61.10666454, 96.51142588],[75.02474557, 46.55401354],[76.0987867, 87.42056972],[84.43281996, 43.53339331],[95.86155507, 38.22527806],[75.01365839, 30.60326323],[82.30705337, 76.4819633 ],[69.36458876, 97.71869196],[39.53833914, 76.03681085],[53.97105215, 89.20735014],[69.07014406, 52.74046973],[67.94685548, 46.67857411],[70.66150955, 92.92713789],[76.97878373, 47.57596365],[67.37202755, 42.83843832],[89.67677575, 65.79936593],[50.53478829, 48.85581153],[34.21206098, 44.2095286 ],[77.92409145, 68.97235999],[62.27101367, 69.95445795],[80.19018075, 44.82162893],[93.1143888, 38.80067034],[61.83020602, 50.25610789],[38.7858038, 64.99568096],[61.37928945, 72.80788731],[85.40451939, 57.05198398],[52.10797973, 63.12762377],[52.04540477, 69.43286012],[40.23689374, 71.16774802],[54.63510555, 52.21388588],[33.91550011, 98.86943574],[64.17698887, 80.90806059],[74.78925296, 41.57341523],[34.18364003, 75.23772034],[83.90239366, 56.30804622],[51.54772027, 46.85629026],[94.44336777, 65.56892161],[82.36875376, 40.61825516],[51.04775177, 45.82270146],[62.22267576, 52.06099195],[77.19303493, 70.4582],[97.77159928, 86.72782233],[62.0730638, 96.76882412],[91.5649745, 88.69629255],[79.94481794, 74.16311935],[99.27252693, 60.999031],[90.54671411, 43.39060181],[34.52451385, 60.39634246],[50.28649612, 49.80453881],[49.58667722, 59.80895099],[97.64563396, 68.86157272],[32.57720017, 95.59854761],[74.24869137, 69.82457123],[71.79646206, 78.45356225],[75.39561147, 85.75993667],[35.28611282, 47.02051395],[56.2538175, 39.26147251],[30.05882245, 49.59297387],[44.66826172, 66.45008615],[66.56089447, 41.09209808],[40.45755098, 97.53518549],[49.07256322, 51.88321182],[80.27957401, 92.11606081],[66.74671857, 60.99139403],[32.72283304, 43.30717306],[64.03932042, 78.03168802],[72.34649423, 96.22759297],[60.45788574, 73.0949981 ],[58.84095622, 75.85844831],[99.8278578, 72.36925193],[47.26426911, 88.475865],[50.4581598, 75.80985953],[60.45555629, 42.50840944],[82.22666158, 42.71987854],[88.91389642, 69.8037889 ],[94.83450672, 45.6943068 ],[67.31925747, 66.58935318],[57.23870632, 59.51428198],[80.366756, 90.9601479 ],[68.46852179, 85.5943071 ],[42.07545454, 78.844786],[75.47770201, 90.424539],[78.63542435, 96.64742717],[52.34800399, 60.76950526],[94.09433113, 77.15910509],[90.44855097, 87.50879176],[55.48216114, 35.57070347],[74.49269242, 84.84513685],[89.84580671, 45.35828361],[83.48916274, 48.3802858 ],[42.26170081, 87.10385094],[99.31500881, 68.77540947],[55.34001756, 64.93193801],[74.775893, 89.5298129 ]])
y_train=np.array([0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0.,1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1.,1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1.,1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1.,1., 1., 1., 1.])

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

value = 0
print (f"sigmoid({value}) = {sigmoid(value)}")
m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)
plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
