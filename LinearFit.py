import numpy as np
import matplotlib.pyplot as plt


def read_file(filename='data.txt'):
    with open(filename, 'r') as ifl:
        array = [l.strip().split('   ') for l in ifl]
        X = np.matrix([float(x) for row in array[2:22] for x in row]).reshape(20, 5)
        b = np.matrix([float(x) for row in array[26:46] for x in row]).reshape(20, 1)
        return X, b


def gradient_descent(A, b, num_iter, rate=0.025,precision=0.0001):
    _, c = A.shape
    #initial matrix value with all of its element as one
    hn = np.matrix(np.ones(c).reshape(c, 1))
    lst =[]
    for i in range(num_iter):
        #Iterative solution to Least square problem from lecture slide number 11
        hn = hn - rate * 2 * A.T * (A * hn - b)
        e = (A * hn - b)
        #finding mean squared error
        error = (e.T * e).tolist()[0][0]
        lst.append(error)
        #checkcing if the mean squared error is less than the precison value to terminate iteration
        if (i > 0) & (abs(lst[i] - lst[i - 1]) < precision):
            break
    return hn,lst


def plot_gradient_descent(error):
    x = range(len(error))
    plt.plot(x, error, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue', markersize=5)
    plt.xlim(0, len(error))
    plt.xlabel('Iterations')
    plt.ylabel('error')
    plt.title('Gradient descent')
    plt.show()


def exact_solution(A, b):
    #Exact solution of equation implementation from lecture slide 8
    return (A.T * A).I * A.T * b

if __name__ == "__main__":
    #Read input file
    X, b = read_file()

    #Iterative solution to LSE method
    h , SquaredError = gradient_descent(X, b, num_iter=100)
    print(' The required h matrix using iterative solution is ')
    print(h)

    plot_gradient_descent(SquaredError)

    # MMSE Exact solution using vector algebra
    print('The exact solution is ', exact_solution(X, b))