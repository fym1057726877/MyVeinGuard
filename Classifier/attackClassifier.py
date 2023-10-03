import os
import torch
import torch.nn as nn
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
import os
import numpy as np

from Classifier.trainClassifier import getDefinedClsModel


def advAttack(classifier, x, label, attack_type, eps):
    if attack_type == "RandFGSM":
        alpha = 0.005
        x = torch.clip(x + alpha * torch.sign(torch.randn(x.shape).to(device)), 0, 1)
        eps2 = eps - alpha
        x_adv = fast_gradient_method(classifier, x, eps2, np.inf)
    elif attack_type == "FGSM":
        x_adv = fast_gradient_method(classifier, x, eps, np.inf)
    elif attack_type == "PGD":
        x_adv = projected_gradient_descent(
            classifier, x,
            eps=eps, eps_iter=1 / 255, nb_iter=min(255 * eps + 4, 1.25 * (eps * 255)), norm=np.inf)
    elif attack_type == "HSJA":
        x_adv = hop_skip_jump_attack(classifier, x, norm=2,
                                     initial_num_evals=1, max_num_evals=50,
                                     num_iterations=30, batch_size=batchsize, verbose=False)
    else:
        raise RuntimeError("The attack type {} is invalid!".format(attack_type))
    return x_adv


def generateAdvImage(classifier, path, attack_type="fgsm"):
    print("Generating Adversarial Examples ...")
    print("eps = {} attack_type = {}".format(eps, attack_type))
    # 加载数据
    data = torch.load("数据路径")
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data, label_data = None, None, None
    loss_fun = nn.CrossEntropyLoss()
    for index, (x, label) in enumerate(data):
        x, label = x.to(device), label.to(device)
        pred = classifier(x)
        train_acc += pred.max(dim=1, keep_dim=True)[1].eq(label.view_as(pred)).sum()

        x_adv = advAttack(classifier=classifier, x=x, label=label, attack_type=attack_type, eps=eps)

        y_adv = classifier(x_adv)
        adv_acc += y_adv.max(dim=1, keep_dim=True)[1].eq(label.view_as(y_adv)).sum()
        train_n += label.size(0)

        x, x_adv, label = x.data, x_adv.data, label.data
        if normal_data is None:
            normal_data, adv_data, label_data = x, x_adv, label
        else:
            normal_data = torch.cat((normal_data, x))
            adv_data = torch.cat((adv_data, x_adv))
            label_data = torch.cat((label_data, label))

    print("Accuracy(normal) {:.6f}, Accuracy(FGSM) {:.6f}".format(torch.true_divide(train_acc, train_n),
                                                                  torch.true_divide(adv_acc, train_n)))

    torch.save({"normal": normal_data, "adv": adv_data, "label": label_data}, path)


if __name__ == "__main__":

    # dataset_name = "Handvein"
    # dataset_name = "Handvein3"
    dataset_name = "Fingervein2"


    model_name = "Resnet18"
    # model_name = "GoogleNet"
    # model_name = "ModelB"
    # model_name = "MSMDGANetCnn_wo_MaxPool"
    # model_name = "Tifs2019Cnn_wo_MaxPool"
    # model_name = "FVRASNet_wo_Maxpooling"
    # model_name = "LightweightDeepConvNN"

    # attack_type = "RandFGSM"
    attack_type = "FGSM"
    # attack_type = "PGD"
    # attack_type = "HSJA"

    device = "cuda"

    batchsize = 32

    if attack_type == "RandFGSM":
        eps = 0.015
    elif attack_type == "PGD":
        # eps = 0.02
        # eps = 0.01
        eps = 0.015
    else:
        # eps = 0.01
        # eps = 0.015
        eps = 0.03
    # eps = 0.015
    # eps = 0.02
    # eps = 0.021
    # eps = 0.023
    # eps = 0.0235
    # eps = 0.024
    # eps = 0.0245
    # eps = 0.0246
    # eps = 0.0247
    # eps = 0.0248
    # eps = 0.0249
    # eps = 0.025
    # eps = 0.03
    # eps = 0.1   # 0.542
    # eps = 0.15  # 0.0.224
    # eps = 0.2   # 0.081

    # eps = 0.3
    # eps = 0.5


    model = getDefinedClsModel(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device
    )
    model.load_state_dict(torch.load("the path of classifier"))
    generateAdvImage(
        classifier=model,
        path=os.path.join(
            "data",
            "AdvData.ckp"
        ),
        attack_type=attack_type
    )