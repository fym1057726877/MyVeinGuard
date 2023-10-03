import torch
from torch import nn
import torch.nn.functional as F
import os

device = 'cuda'

class DeAe1Model(nn.Module):
    def __init__(self, img_shape):
        super(DeAe1Model, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2),  
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        return x

class DenoisingAutoEncoder_1:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool2d(2),   
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),   
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid()).to(device)

    def train(self, dataloader, lr, v_noise=0, num_epochs=100):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            
            self.model.train()
            
            epoch_loss = 0
            for batchIndex, (image, label) in enumerate(dataloader):
                
                noise = v_noise * torch.randn_like(image)
                
                noisy_data = torch.clamp(image + noise, min=0, max=1)
                
                image = torch.autograd.Variable(image).to(device)
                noisy_data = torch.autograd.Variable(noisy_data).to(device)

                optimizer.zero_grad()
                
                output = self.model(noisy_data)
                
                loss = F.mse_loss(output, image)
                loss.backward()
                optimizer.step()

                
                epoch_loss = epoch_loss + float(loss)
                
                
            
            
            print("Train Epoch: %d [Loss: %6f]" % (epoch, epoch_loss))
        
        
        save_path = os.path.join(
            "MagNet", "AeModel1.ckp"
        )
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
        self.model.eval()



class DeAe2Model(nn.Module):
    def __init__(self, img_shape):
        super(DeAe2Model, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DenoisingAutoEncoder_2:
    def __init__(self, img_shape):
        self.img_shape = img_shape

        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid()).to(device)

    def train(self, dataloader, lr, v_noise=0, num_epochs=100):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        
        for epoch in range(num_epochs):
            self.model.train()
            batch_loss = 0
            for batch_i, (data_train, data_label) in enumerate(dataloader):
                noise = v_noise * torch.randn_like(data_train)
                noisy_data = torch.clamp(data_train + noise, min=0, max=1)
                data_train = torch.autograd.Variable(data_train).to(device)
                noisy_data = torch.autograd.Variable(noisy_data).to(device)

                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = F.mse_loss(output, data_train)
                loss.backward()
                batch_loss = batch_loss + float(loss)
                optimizer.step()
                
                
            print("Train Epoch: %d [Loss: %6f]" % (epoch, batch_loss))
        
        save_path = os.path.join(
            "MagNet", "AeModel2.ckp"
        )
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
        self.model.eval()

