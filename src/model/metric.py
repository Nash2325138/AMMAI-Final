import torch


class TopKAcc():
    def __init__(self, k, nickname="", output_key='verb_logits', target_key='verb_class'):
        self.k = k
        self.__name__ = f'top{self.k}_acc_{target_key}' if nickname == "" else nickname
        self.output_key = output_key
        self.target_key = target_key

    def __call__(self, data, output):
        with torch.no_grad():
            logits = output[self.output_key]
            target = data[self.target_key]
            pred = torch.topk(logits, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
