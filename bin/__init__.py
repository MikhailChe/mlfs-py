import numpy as np
from matplotlib import pyplot as plt

from library import Line, Relu


def main():
    IL = 32

    layers = [
        Line(weight=(np.random.rand(IL, 1) - .5), bias=np.random.rand(IL) * 10.),
        Relu(),
        Line(weight=(np.random.rand(IL, IL) - .5), bias=np.random.rand(IL) * 10.),
        Relu(),
        Line(weight=(np.random.rand(1, IL) - .5), bias=np.random.rand(1) * 10.),
    ]

    plt.ion()
    Xs = np.linspace(-10, 10, 128)
    Ys = np.sin(Xs * .6) + np.random.rand(Xs.shape[0]) * .2

    Xs = np.array([Xs]).T
    Ys = np.array([Ys]).T
    plt.scatter(Xs, Ys)

    PXs = np.array(np.array([np.linspace(np.min(Xs)-5, np.max(Xs)+5, 128)]).T)
    PYs = np.zeros(PXs.shape)

    err_text = plt.text(Xs[0], Ys[0], 'ERR')
    prediction, = plt.plot(PXs, PYs)
    for i in range(10000000):
        if i % 10000 == 0:
            prediction_tensors = [PXs]
            for layer in layers:
                last_tensor = prediction_tensors[-1]
                prediction_tensors.append(layer.forward(last_tensor))
            prediction.set_ydata(prediction_tensors[-1])
            plt.pause(.0001)

        tensors = [Xs]
        for layer in layers:
            last_tensor = tensors[-1]
            tensors.append(layer.forward(last_tensor))

        diff = tensors[-1] - Ys
        err = np.mean(diff ** 2)
        err_text.set_text(f'{err}')

        nabla = diff
        for i, layer in reversed(list(enumerate(layers))):
            nabla = layer.backward(tensors[i], nabla)


if __name__ == '__main__':
    main()
