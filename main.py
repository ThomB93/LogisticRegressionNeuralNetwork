import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize
from lr_utils import load_dataset
from skimage import transform

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("To be or not to be (a cat)")

st.write("Let's train a neural network using Logistic Regression to predict whether an image contains a cat!")
st.write("Our training data consists of 208 images - with some cats in between.")
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = st.number_input("Input an index to see the images we use to train the model!", step=1, min_value=0, max_value=208, value=0)
st.image(train_set_x_orig[index], width=250)

m_train = train_set_x_orig.shape[0]
st.write("Number of training samples: %s" % m_train)
m_test = test_set_x_orig.shape[0]
st.write("Number of testing samples: %s" % m_test)
num_px = train_set_x_orig.shape[1]
st.write("Number of pixels (height and width): %s" % num_px)

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

st.title("Forward Propagation")
st.write("First, let's feed the data into the neural network to calculate the values of the activation functions and finally the cost.")
st.write("Here we will be using the sigmoid activation function, which will output a value between 0 and 1.")
st.write("The activation is calculated as follows:")
r'''
$$A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$$
'''
st.write("The logistic regression cost is calculated as follows:")
r'''
$$J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$$
'''

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


dim = 2
w, b = initialize_with_zeros(dim)


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation

    cost = -(1 / m) * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}
    return grads, cost


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = None
    db = None
    with st.empty():
        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                st.write("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


#params, grads, costs = optimize(w, b, X, Y, num_iterations= num_iterations, learning_rate = learning_rate, print_cost = True)

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    print(A)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = A[0, i] >= 0.5

    return Y_prediction


w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])


@st.cache(suppress_st_warning=True)
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    st.write("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    st.write("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


st.write("Change the values for the number of iterations and the learning rate, to see their effect on the training and test accuracy. Note that any change will retrain the model.")
num_iterations = st.number_input("Input number of iterations (0-5000):", value=0, max_value=5000, min_value=0, step=10)
learning_rate = st.slider("Select learning rate:", min_value=0.0, max_value=0.1, value=0.0, step=0.01)

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = num_iterations, learning_rate = learning_rate, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
st.pyplot()

st.title("Predict")
st.write("With the model trained, let's make some predictions!")
st.write("Upload an image below to have the model make a prediction.")

uploaded_image = st.file_uploader("Select an image to upload", type=['png','jpeg', 'jpg'])
if uploaded_image is not None:
    file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type,"FileSize":uploaded_image.size}
    st.write(file_details)
    image = np.array(imageio.imread(uploaded_image))

    image = image / 255.
    #my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_image = resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)


st.write("y = " + str(np.squeeze(my_predicted_image)) + ", the model predicts that your image is a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")