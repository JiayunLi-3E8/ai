import numpy as np
import pickle


class Neuron:
    def __init__(self, inputNum: int):
        weights = []
        for i in range(inputNum):
            weights.append(np.random.normal())
        self._weights = np.array(weights)
        self._bias = np.random.normal()

    @staticmethod
    def _sigmoid(x) -> np.float64:
        return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

    @classmethod
    def _dNormalize(cls, x, limit=1) -> np.float64:
        return (cls._sigmoid(1 / x) - 0.5) * 2 * limit if x != 0 else limit

    @classmethod
    def _activation(cls, x) -> np.float64:
        return cls._sigmoid(x)

    @classmethod
    def _activation_deriv(cls, x) -> np.float64:
        fx = cls._sigmoid(x)
        return fx * (1 - fx)

    def _basis(self, inputs: np.ndarray) -> np.float64:
        return np.dot(self._weights, inputs) + self._bias

    def feedforward(self, inputs: np.ndarray):
        total = self._basis(inputs)
        return self._activation(total)

    def backpropagation(self, inputs: np.ndarray, learnRate: float, lastDerivBack: np.float64 = 1) -> np.ndarray:
        d = self._activation_deriv(self._basis(inputs)) * lastDerivBack
        back = d * self._weights
        for idx in range(self._weights.size):
            #     self._weights[idx] -= learnRate * self._dNormalize(d * inputs[idx])
            # self._bias -= learnRate * self._dNormalize(d)
            self._weights[idx] -= learnRate * d * inputs[idx]
        self._bias -= learnRate * d
        return back


class NeuronNetwork:
    def __init__(self, inputNum: int, hiddenNeuronNum: int, initLearnRate: float = 1):
        self._learnRate = initLearnRate
        self._hidden = []
        for i in range(hiddenNeuronNum):
            self._hidden.append(Neuron(inputNum))
        self._outlet = Neuron(hiddenNeuronNum)

    @staticmethod
    def _loss(preds: np.ndarray, trues: np.ndarray) -> np.float64:
        # 均方误差（MSE）
        return ((preds - trues) ** 2).mean()

    @staticmethod
    def _loss_deriv(preds: np.ndarray, trues: np.ndarray):
        assert preds.size == trues.size
        d = []
        for idx in range(trues.size):
            d.append((preds[idx] - trues[idx]) * 2 / trues.size)
        return np.array(d)

    def feedforward(self, inputs: np.ndarray):
        hiddenOuts = []
        for h in self._hidden:
            assert isinstance(h, Neuron)
            hiddenOuts.append(h.feedforward(inputs))
        return self._outlet.feedforward(np.array(hiddenOuts))

    def trainOnceData(self, data: np.ndarray, dl: float):
        hiddenOuts = []
        for h in self._hidden:
            assert isinstance(h, Neuron)
            hiddenOuts.append(h.feedforward(data))
        transfer = self._outlet.backpropagation(
            hiddenOuts, self._learnRate, dl)
        assert transfer.size == len(self._hidden)
        for idx, hidNeu in enumerate(self._hidden):
            assert isinstance(hidNeu, Neuron)
            hidNeu.backpropagation(data, self._learnRate, transfer[idx])

    def train(self, datas: np.ndarray, trues: np.ndarray):
        preds = []
        for data in datas:
            assert isinstance(data, np.ndarray)
            preds.append(self.feedforward(data))
        dls = self._loss_deriv(np.array(preds), trues)
        for idx in range(dls.size):
            self.trainOnceData(datas[idx], dls[idx])
        preds.clear()
        for data in datas:
            assert isinstance(data, np.ndarray)
            preds.append(self.feedforward(data))
        return self._loss(np.array(preds), trues)

    def trainForTarget(self, datas: np.ndarray, trues: np.ndarray, TargetLoss: np.float64):
        lastLoss = 1
        i = 0
        while lastLoss > TargetLoss and self._learnRate != 0:
            i += 1
            afterLoss = self.train(datas, trues)
            if not i % 1000:
                print("after {}\ttrain,loss {}\t,rate {}".format(
                    i, afterLoss, self._learnRate))
            if afterLoss - lastLoss >= 0:
                self._learnRate /= 2
                print("after {}\ttrain,loss {}\t,rate {}".format(
                    i, afterLoss, self._learnRate))
            # elif afterLoss - lastLoss > -0.000001:
            #     self._learnRate += 0.01
            #     print("after {}\ttrain,loss {}\t,rate {}".format(
            #         i, afterLoss, self._learnRate))
            lastLoss = afterLoss
        print("after {}\ttrain,loss {}\t,rate {}".format(
            i, afterLoss, self._learnRate))


def main():
    # datas = np.array([
    #     [-2, -1],  # Alice
    #     [25, 6],  # Bob
    #     [17, 4],  # Charlie
    #     [-15, -6]  # diana
    # ])
    datas = np.array([
        [162, 48],  # Alice
        [178, 70],  # Bob
        [182, 86],  # Charlie
        [158, 52]  # diana
    ])
    all_y_trues = np.array([
        0,  # Alice
        1,  # Bob
        1,  # Charlie
        0  # diana
    ])
    try:
        objFile = open("NNB", "rb")
    except FileNotFoundError:
        net = NeuronNetwork(2, 64)
        net.trainForTarget(datas, all_y_trues, 0.0001)
        objFile = open("NNB", "wb")
        pickle.dump(net, objFile)
        objFile.close()
    else:
        net = pickle.load(objFile)
        objFile.close()
        assert isinstance(net, NeuronNetwork)

        # net._learnRate = 2
        # net.trainForTarget(datas, all_y_trues, 0.0001)
        # objFile = open("NNB", "wb")
        # pickle.dump(net, objFile)
        # objFile.close()

    while True:
        inp = eval(input("input like (a,b):"))
        print(net.feedforward(np.array(inp)))


if(__name__ == "__main__"):
    main()
