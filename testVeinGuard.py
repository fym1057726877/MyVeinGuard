import torch
from Classifier.trainClassifier import getDefinedClsModel
from DirectActor.ActorModel import Resnet34Actor
from TransGAN.TransGanModel import SwinTransGenerator


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"

actor_type = "ResnetActor"

is_local = True
# is_local = False
is_peg = True
# is_peg = False

device = "cuda"
batch_size = 24

def getAdvDataLoader(attack_type, eps, model_name, genAdvExampleModel_name):
    # customize this function according to your data path of adv data
    pass


class DirectTrainActorReconstruct:
    def __init__(self):
        super(DirectTrainActorReconstruct, self).__init__()
        self.actor = Resnet34Actor().to(device)
        self.actor.load_state_dict(torch.load("path of actor model"))
        self.actor.to(device)

        self.generator = SwinTransGenerator(is_local=is_local, is_peg=is_peg)
        self.generator.load_state_dict(torch.load("path of g model"))
        self.generator.to(device)


    @torch.no_grad()
    def defense(self, img):
        z_tmp = self.actor(img)
        rec_img = self.generator(z_tmp)
        return rec_img

    @torch.no_grad()
    def test(self, attack_type, eps, model_name, genAdvExampleModel_name):
        classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device
        )
        classifier.load_state_dict(torch.load("path of classifier"))
        advDataloader = getAdvDataLoader(attack_type, eps, model_name, genAdvExampleModel_name)
        total_num = len(advDataloader)

        def accuracy(y, y1):
            return y.max(dim=1, keep_dim=True)[1].eq(y1.view_as(y)).sum()

        normal_acc, adv_acc, rec_acc, num = 0, 0, 0, 0
        iterObject = enumerate(advDataloader)
        for i, (img, adv_img, label) in iterObject:
            img, adv_img = img.to(self.device), adv_img.to(self.device)
            label = label.to(self.device)
            y = classifier(img)
            normal_acc += accuracy(y, label)

            adv_y = classifier(adv_img)
            adv_acc += accuracy(adv_y, label)
            rec_img = self.defense(adv_img)

            rec_y = classifier(rec_img)
            rec_acc += accuracy(rec_y, label)
            num += label.size(0)

        print("[Num: %d/%d] [NormalAcc:   %f] [AdvAcc:   %f] [RecAcc:   %f]" % (
            num, total_num,
            torch.true_divide(normal_acc, num).item(),
            torch.true_divide(adv_acc, num).item(),
            torch.true_divide(rec_acc, num).item()))


def testDirectActor():
    # seed = 1  # the seed for random function

    attack_type = 'FGSM'
    # attack_type = 'PGD'
    # attack_type = 'RandFGSM'
    # attack_type = 'HSJA'
    if attack_type == 'RandFGSM':
        eps = 0.015
    elif attack_type == "PGD":
        # eps = 0.01
        eps = 0.015
    else:
        # eps = 0.02
        eps = 0.01

    model_name = "Resnet18"
    # model_name = "GoogleNet"
    # model_name = "ModelB"
    # model_name = "MSMDGANetCnn_wo_MaxPool"
    # model_name = "Tifs2019Cnn_wo_MaxPool"
    # model_name = "FVRASNet_wo_Maxpooling"
    # model_name = "LightweightDeepConvNN"


    directActorRec = DirectTrainActorReconstruct()

    directActorRec.test(
        attack_type=attack_type,
        eps=eps,
        model_name=model_name,
        genAdvExampleModel_name=model_name
    )


if __name__ == "__main__":
    testDirectActor()
