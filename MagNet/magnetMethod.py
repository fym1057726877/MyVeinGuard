
import torch
import numpy as np



class AEDectector:
    def __init__(self, model, load_path=None, p=1, device="cuda"):
        if load_path is not None:
            model.load(load_path)
        self.model = model.model
        self.path = load_path
        self.p = p
        self.device = device

    def mark(self, image):
        image = image.to(self.device)
        diff = torch.abs(image - self.model(image))
        marks = torch.mean(torch.pow(diff, self.p), dim=(1, 2, 3))
        return marks


class SimpleReformer:
    def __init__(self, model, load_path=None):
        self.model = model.model
        self.path = load_path

    def heal(self, image):
        image = self.model(image)
        return torch.clamp(image, 0.0, 1.0)


class Operator:
    def __init__(self, normalData, classifier, detector_dict, reformer, device="cuda"):
        self.device = device
        self.normalData = normalData
        self.classifier = classifier
        self.detector_dict = detector_dict
        self.reformer = reformer
        self.normal = self.operate(self.normalData)

    def operate(self, untrusted_obj):
        image = untrusted_obj.image.to(self.device)
        label = untrusted_obj.label.to(self.device)
        image_prime = self.reformer.heal(image)
        pred_label = torch.argmax(self.classifier(image), dim=1)
        label_judge = (pred_label == label[:len(image_prime)])
        pred_label_prime = torch.argmax(self.classifier(image_prime), dim=1)
        label_judge_prime = (pred_label_prime == label[:len(image_prime)])
        print('label_judge', label_judge)
        print('label_judge_prime', label_judge_prime)
        return np.array(list(zip(label_judge.cpu(), label_judge_prime.cpu())))

    def filter(self, X, thrs):
        collector = dict()
        all_pass = np.array(range(10000))
        for name, detector in self.detector_dict.items():
            marks = detector.mark(X).cpu().detach().numpy()
            idx_pass = np.argwhere(marks < thrs[name].cpu().detach().numpy())
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)

    def get_thrs(self, drop_rate):
        thrs = dict()
        for name, detector in self.detector_dict.items():
            num = int(len(self.normalData) * drop_rate[name])
            marks = detector.mark(self.normalData.image)
            marks, _ = torch.sort(marks)
            thrs[name] = marks[-num]
        return thrs


class Evauator():
    def __init__(self, operator, untrusted_data):
        self.operator = operator
        self.untrusted_data = untrusted_data
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):
        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        filtered_normal_tups = normal_tups[normal_all_pass]
        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC) / num_normal
        det_only_acc = sum(1 for XC, XpC in filtered_normal_tups if XC) / num_normal
        ref_only_acc = sum([1 for _, XpC in normal_tups if XpC]) / num_normal
        none_acc = sum([1 for XC, _ in normal_tups if XC]) / num_normal
        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)
        filtered_attack_tups = attack_tups[attack_pass]
        both_num = sum(1 for _, XpC in filtered_attack_tups if XpC)
        both_acc = sum(1 for _, XpC in filtered_attack_tups if XpC) / num_untrusted
        det_only_acc = sum(1 for XC, XpC in filtered_attack_tups if XC) / num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups if XpC]) / num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups if XC]) / num_untrusted
        return both_acc, det_only_acc, ref_only_acc, none_acc