from Classifier.trainClassifier import getDefinedClsModel
import os

from MagNet.magnetMethod import AEDectector, SimpleReformer, Operator, Evauator
from MagNet.model import DenoisingAutoEncoder_1, DenoisingAutoEncoder_2

import torch
# customize these object according to the path of your own data
import getOriginalTensorData



class NormalDataList:
    def __init__(self):
        self.image = None
        self.label = None

    def __len__(self):
        return len(self.image)


class AttackDataList:
    def __init__(self):
        self.image = None
        self.label = None

    def __len__(self):
        return len(self.image)


class Worker:
    def __init__(
            self,
            dataset_name,
            model_name,
            genAdvExampleModel_name,
            attack_type,
            eps,
            device
    ):
        with torch.no_grad():
            
            clf = getDefinedClsModel(
                dataset_name=dataset_name,
                model_name=model_name,
                device=device
            )
            clf.load_state_dict(torch.load("the path of classifier"))
            dataTuple = getOriginalTensorData(dataset_name)
            normalData = NormalDataList()
            normalData.image = dataTuple["test_image"]
            normalData.label = dataTuple["test_label"]
            # customize these object according to the path of your own data
            attack_data_path = os.path.join(
                f"AdvTestImg_{genAdvExampleModel_name}_{attack_type}_{eps}.ckp"
            )
            attack_dataTuple = torch.load(attack_data_path)
            self.adversarialData = AttackDataList()
            self.adversarialData.image = attack_dataTuple["adv"]
            self.adversarialData.label = attack_dataTuple["label"]
            
            
            if dataset_name == "Fingervein1":
                img_shape = (1, 64, 128)
            else:
                img_shape = (1, 64, 64)
            dae1 = DenoisingAutoEncoder_1(img_shape=img_shape)
            dae1.load(
                load_path=os.path.join(
                    "MagNet", "AeModel1.ckp"
                )
            )
            
            dae2 = DenoisingAutoEncoder_2(img_shape=img_shape)
            dae2.load(
                load_path=os.path.join(
                    "MagNet", "AeModel2.ckp"
                )
            )
            detector_1 = AEDectector(dae1, p=2)
            detector_2 = AEDectector(dae2, p=1)
            detector_dict = dict()
            detector_dict["1"] = detector_1
            detector_dict["2"] = detector_2
            reformer = SimpleReformer(dae1)
            operator = Operator(normalData, clf, detector_dict, reformer)
            self.evau = Evauator(operator, self.adversarialData)


    @torch.no_grad()
    def work_hard(self):
        thrs = self.evau.operator.get_thrs({'1': 0.001, '2': 0.001})
        attack_all_pass = self.evau.operator.filter(self.adversarialData.image, thrs)[0]
        result = self.evau.get_attack_acc(attack_all_pass)
        print('No defense accuracy : ', result[3])
        print('Reformer only accuracy : ', result[2])
        print('Detector only accuracy : ', result[1])
        print('Both detector and reformer accuracy : ', result[0])


def work(dataset_name, model_name, genAdvExampleModel_name, attack_type, eps, device):
    clf = getDefinedClsModel(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device
    )
    clf.load_state_dict(torch.load("the path of classifier"))
    dataTuple = getOriginalTensorData(dataset_name)
    normalData = NormalDataList()
    normalData.image = dataTuple["test_image"]
    normalData.label = dataTuple["test_label"]
    
    adversarialData = AttackDataList()
    # customize these object according to the path of your own data
    attack_data_path = os.path.join(
        f"AdvTestImg_{genAdvExampleModel_name}_{attack_type}_{eps}.ckp"
    )
    attack_dataTuple = torch.load(attack_data_path)
    adversarialData.image = attack_dataTuple["adv"]
    adversarialData.label = attack_dataTuple["label"]
    if dataset_name == "Fingervein1":
        img_shape = (1, 64, 128)
    else:
        img_shape = (1, 64, 64)
    dae1 = DenoisingAutoEncoder_1(img_shape=img_shape)
    dae1.load(
        load_path=os.path.join(
            "MagNet", "ae_model1.ckp"
        )
    )
    
    dae2 = DenoisingAutoEncoder_2(img_shape=img_shape)
    dae1.load(
        load_path=os.path.join(
            "MagNet", "ae_model2.ckp"
        )
    )

    detector_1 = AEDectector(dae1, p=2)
    detector_2 = AEDectector(dae2, p=1)
    
    detector_dict = dict()
    detector_dict["1"] = detector_1
    detector_dict["2"] = detector_2
    reformer = SimpleReformer(dae1)
    operator = Operator(normalData, clf, detector_dict, reformer)
    evau = Evauator(operator, adversarialData)
    thrs = evau.operator.get_thrs({'1': 0.001, '2': 0.001})
    attack_all_pass = evau.operator.filter(adversarialData.image, thrs)[0]
    result = evau.get_attack_acc(attack_all_pass)

    print('No defense accuracy : ', result[3])
    print('Reformer only accuracy : ', result[2])
    print('Detector only accuracy : ', result[1])
    print('Both detector and reformer accuracy : ', result[0])

def whiteBox_andHsja():
    device = 'cuda'

    # dataset_name = "Handvein"
    # dataset_name = "Handvein3"
    dataset_name = "Fingervein1"

    model_name = "Resnet18"
    # model_name = "GoogleNet"
    # model_name = "ModelB"
    # model_name = "MSMDGANetCnn_wo_MaxPool"
    # model_name = "Tifs2019Cnn_wo_MaxPool"
    # model_name = "FVRASNet_wo_Maxpooling"
    # model_name = "LightweightDeepConvNN"

    genAdvExampleModel_name = "Resnet18"
    # genAdvExampleModel_name = "GoogleNet"
    # genAdvExampleModel_name = "ModelB"
    # genAdvExampleModel_name = "MSMDGANetCnn_wo_MaxPool"
    # genAdvExampleModel_name = "Tifs2019Cnn_wo_MaxPool"
    # genAdvExampleModel_name = "FVRASNet_wo_Maxpooling"
    # genAdvExampleModel_name = "LightweightDeepConvNN"

    eps = 0.01
    # eps = 0.015
    # eps = 0.02
    # # eps = 0.023
    # # eps = 0.0235
    # # eps = 0.024
    # # eps = 0.0245
    # # eps = 0.0246
    # # eps = 0.0247
    # # eps = 0.0248
    # # eps = 0.0249
    # eps = 0.025
    # # eps = 0.03

    # attack_type = "RandFGSM"
    attack_type = "FGSM"
    # attack_type = "PGD"
    # attack_type = "HSJA"

    if attack_type == "RandFGSM":
        eps = 0.015

    worker = Worker(
        dataset_name=dataset_name,
        model_name=model_name,
        genAdvExampleModel_name=model_name,
        attack_type=attack_type,
        eps=eps,
        device=device
    )
    worker.work_hard()


def blackBox_transfer():
    device = 'cuda'

    dataset_name = "Handvein"
    # dataset_name = "Handvein3"
    # dataset_name = "Fingervein1"

    model_name = "Resnet18"
    # model_name = "GoogleNet"
    # model_name = "ModelB"
    # model_name = "MSMDGANetCnn_wo_MaxPool"
    # model_name = "Tifs2019Cnn_wo_MaxPool"
    # model_name = "FVRASNet_wo_Maxpooling"
    # model_name = "LightweightDeepConvNN"

    genAdvExampleModel_name = "Resnet18"
    # genAdvExampleModel_name = "GoogleNet"
    # genAdvExampleModel_name = "ModelB"
    # genAdvExampleModel_name = "MSMDGANetCnn_wo_MaxPool"
    # genAdvExampleModel_name = "Tifs2019Cnn_wo_MaxPool"
    # genAdvExampleModel_name = "FVRASNet_wo_Maxpooling"
    # genAdvExampleModel_name = "LightweightDeepConvNN"

    eps = 0.01
    # # eps = 0.015
    # # eps = 0.02
    # # eps = 0.023
    # # eps = 0.0235
    # # eps = 0.024
    # # eps = 0.0245
    # # eps = 0.0246
    # # eps = 0.0247
    # # eps = 0.0248
    # # eps = 0.0249
    # eps = 0.025
    # # eps = 0.03

    # attack_type = "RandFGSM"
    # attack_type = "FGSM"
    attack_type = "PGD"

    work(
        dataset_name=dataset_name,
        model_name=model_name,
        genAdvExampleModel_name=genAdvExampleModel_name,
        attack_type=attack_type,
        eps=eps,
        device=device
    )

def main():
    whiteBox_andHsja()
    # blackBox_transfer()

if __name__ == "__main__":
    main()
