import numpy as np

import matplotlib.pyplot as plt


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)

def task_10_quiz_softmax():
    scores = np.array([3.0, 1.0, 0.2])
    print(softmax(scores))
    print(np.sum(softmax(scores)))
    print(softmax(scores * 10))

    # Plot softmax curves

    x = np.arange(-3.0, 6.0, 0.01)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    plt.plot(x, softmax(scores).T, linewidth=2)
    plt.show()

def task_18_precision():
    big_number = 1e9
    small_number = 1e-6
    x = big_number
    for _ in range(1000 * 1000):
        x += small_number
    print(x)
    x -= big_number
    print(x)

def main():
    task_18_precision()


if __name__ == '__main__':
    main()