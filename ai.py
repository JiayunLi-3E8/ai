import numpy as np
import pickle
from neuronNetwork import NeuronNetwork


def main():
    datas = np.array([
        [83124], [145118], [873696], [907834],
        [323634], [928608], [1725483], [1899408],
        [2380625]
    ]) * 1e-6 - 1
    all_y_trues = np.array([
        676527, 242137, 253016, 586961,
        773726, 218453, 542263, 378339,
        792874
    ])* 1e-6
    try:
        objFile = open("NNB", "rb")
    except FileNotFoundError:
        net = NeuronNetwork(1, 16, 2, 0.5)
        net.trainForTarget(datas, all_y_trues, 0.045)
        objFile = open("NNB", "wb")
        pickle.dump(net, objFile)
        objFile.close()
    else:
        net = pickle.load(objFile)
        objFile.close()
        assert isinstance(net, NeuronNetwork)
        net.trainForTarget(datas, all_y_trues, 1e-8)
        objFile = open("NNB", "wb")
        pickle.dump(net, objFile)
        objFile.close()

    while True:
        inp = (eval(input("id: ")) * 1e-6 - 1,)
        otp = net.feedforward(np.array(inp))
        result = int(otp * 1e6)
        print(result)


if(__name__ == "__main__"):
    main()
