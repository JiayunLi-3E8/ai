import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from neuron import Neuron


class NeuronNetwork:
    def __init__(self, inputNum: int, hiddenNeuronNum: int, hiddenLayerNum: int, initLearnRate: float = 0.5):
        self._inputNum = inputNum
        self._hiddenNeuronNum = hiddenNeuronNum
        self._hiddenLayerNum = hiddenLayerNum
        self._initLearnRate = initLearnRate
        self._init()

    def _init(self):
        self._learnRate = self._initLearnRate
        hiddenLayer = []
        for i in range(self._hiddenNeuronNum):
            hiddenLayer.append(Neuron(self._inputNum))
        self._hiddenLayers = [hiddenLayer]
        for n in range(self._hiddenLayerNum-1):
            hiddenLayer = []
            for i in range(self._hiddenNeuronNum):
                hiddenLayer.append(Neuron(self._hiddenNeuronNum))
            self._hiddenLayers.append(hiddenLayer)
        self._outlet = Neuron(self._hiddenNeuronNum)

    def _loss(self, datas: np.ndarray, trues: np.ndarray) -> np.float64:
        preds = np.apply_along_axis(self.feedforward, 1, datas)
        # 均方误差（MSE）
        return ((preds - trues) ** 2).mean()

    def _loss_deriv(self, datas: np.ndarray, trues: np.ndarray):
        preds = np.apply_along_axis(self.feedforward, 1, datas)
        d = []
        for idx in range(trues.size):
            # MSE_deriv
            d.append((preds[idx] - trues[idx]) * 2 / trues.size)
        return np.array(d)

    def feedforward(self, inputs: np.ndarray):
        hiddenLayerOut = []
        for hNeuron in self._hiddenLayers[0]:
            assert isinstance(hNeuron, Neuron)
            hiddenLayerOut.append(hNeuron.feedforward(inputs))
        for hiddenLayer in self._hiddenLayers[1:]:
            hOuts = []
            for hNeuron in hiddenLayer:
                assert isinstance(hNeuron, Neuron)
                hOuts.append(hNeuron.feedforward(np.array(hiddenLayerOut)))
            hiddenLayerOut = hOuts
        return self._outlet.feedforward(np.array(hiddenLayerOut))

    def _backpropagation(self, hiddenLayer: list, backInputData: np.ndarray, transfers):
        nextTransfers = []
        if isinstance(transfers, np.ndarray):
            for idx, hidNeu in enumerate(hiddenLayer):
                assert isinstance(hidNeu, Neuron)
                nextTransfers.append(hidNeu.backpropagation(
                    backInputData, self._learnRate, transfers[idx]))
        elif isinstance(transfers, list):
            for transfer in transfers:
                nextTransfers.append(self._backpropagation(
                    hiddenLayer, backInputData, transfer))
        return nextTransfers

    def trainOnceData(self, data: np.ndarray, dl: float):
        backInputDatas = [data]
        hiddenLayerOut = []
        for hNeuron in self._hiddenLayers[0]:
            assert isinstance(hNeuron, Neuron)
            hiddenLayerOut.append(hNeuron.feedforward(data))
        backInputDatas.append(np.array(hiddenLayerOut))
        for hiddenLayer in self._hiddenLayers[1:]:
            hOuts = []
            for hNeuron in hiddenLayer:
                assert isinstance(hNeuron, Neuron)
                hOuts.append(hNeuron.feedforward(np.array(hiddenLayerOut)))
            backInputDatas.append(np.array(hOuts))
            hiddenLayerOut = hOuts
        transfer = self._outlet.backpropagation(
            backInputDatas[-1], self._learnRate, dl)
        transfers = [transfer]
        for idx, hiddenLayer in enumerate(self._hiddenLayers[::-1]):
            transfers = self._backpropagation(
                hiddenLayer, backInputDatas[-(idx+2)], transfers)

    def train(self, datas: np.ndarray, trues: np.ndarray):
        dls = self._loss_deriv(datas, trues)
        for idx in range(len(datas)):
            self.trainOnceData(datas[idx], dls[idx])
        return self._loss(datas, trues)

    def trainForTarget(self, datas: np.ndarray, trues: np.ndarray, TargetLoss: np.float64) -> np.float64:
        plt.ion()
        plt.figure(1)
        lastLoss = 0.3
        i = 0
        slow = False
        while lastLoss > TargetLoss and not slow:
            lastRate = self._learnRate
            afterLoss = self.train(datas, trues)
            i += 1
            slow = math.isclose(afterLoss, lastLoss, rel_tol=1e-10)
            # if math.isclose(afterLoss, lastLoss, rel_tol=1e-9):
            #     self._init()
            # elif math.isclose(afterLoss, lastLoss, rel_tol=1e-7):
            #     self._learnRate = 1.005 * self._learnRate
            # elif afterLoss - lastLoss > 0.01:
            #     self._learnRate = self._initLearnRate
            plt.plot((i-1, i), (lastLoss, afterLoss), color='r')
            plt.plot((i-1, i), (lastRate, self._learnRate), color='b')
            plt.xlabel('loss:{:<.18f}  rate:{}'.format(
                afterLoss, self._learnRate))
            plt.pause(sys.float_info.min)
            lastLoss = afterLoss
        return lastLoss
