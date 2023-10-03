
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=256):
        super(ConvGenerator, self).__init__()
        self.net_dim = 48
        self.init_size = 4
        self.fc = nn.Linear(latent_dim, self.init_size * self.init_size * 4 * self.net_dim)
        self.bn = nn.BatchNorm1d(self.init_size * self.init_size * 4 * self.net_dim)
        self.relu = nn.ReLU()

        self.convT1 = nn.ConvTranspose2d(in_channels=4*self.net_dim, out_channels=2*self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(2*self.net_dim)
        self.relu1 = nn.ReLU()

        self.convT2 = nn.ConvTranspose2d(in_channels=2 * self.net_dim, out_channels=self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(self.net_dim)
        self.relu2 = nn.ReLU()

        self.convT3 = nn.ConvTranspose2d(in_channels=self.net_dim, out_channels=self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)

        self.convT4 = nn.ConvTranspose2d(in_channels=self.net_dim, out_channels=1,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)

        self.out = nn.Sigmoid()

    def forward(self, noise):
        x = self.fc(noise)
        x = self.relu(x)
        x = x.reshape(-1, 4*self.net_dim, self.init_size, self.init_size)  
        x = self.convT1(x)   
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.convT2(x)  
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.convT3(x)  
        x = self.convT4(x)
        x = self.out(x)
        return x


class ConvGeneratorFingervein1(nn.Module):
    def __init__(self, latent_dim=256, init_h=4, init_w=8):
        super(ConvGeneratorFingervein1, self).__init__()
        self.net_dim = 48
        self.init_h = init_h
        self.init_w = init_w
        self.fc = nn.Linear(latent_dim, self.init_h * self.init_w * 4 * self.net_dim)
        self.bn = nn.BatchNorm1d(self.init_h * self.init_w * 4 * self.net_dim)
        self.relu = nn.ReLU()

        self.convT1 = nn.ConvTranspose2d(in_channels=4*self.net_dim, out_channels=2*self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(2*self.net_dim)
        self.relu1 = nn.ReLU()

        self.convT2 = nn.ConvTranspose2d(in_channels=2 * self.net_dim, out_channels=self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(self.net_dim)
        self.relu2 = nn.ReLU()

        self.convT3 = nn.ConvTranspose2d(in_channels=self.net_dim, out_channels=self.net_dim,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)

        self.convT4 = nn.ConvTranspose2d(in_channels=self.net_dim, out_channels=1,
                                         kernel_size=5, stride=2, padding=2, output_padding=1)
        self.out = nn.Sigmoid()

    def forward(self, noise):
        x = self.fc(noise)
        x = self.relu(x)
        x = x.reshape(-1, 4*self.net_dim, self.init_h, self.init_w)   
        
        x = self.convT1(x)   
        x = self.bn1(x)
        x = self.relu1(x)
        

        x = self.convT2(x)  
        x = self.bn2(x)
        x = self.relu2(x)
        

        x = self.convT3(x)  
        

        x = self.convT4(x)  
        

        x = self.out(x)
        return x


class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        self.net_dim = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.net_dim, kernel_size=5, stride=2)
        self.leakyRelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=self.net_dim, out_channels=2*self.net_dim, kernel_size=5, stride=2)
        self.leakyRelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(in_channels=2*self.net_dim, out_channels=4 * self.net_dim, kernel_size=5, stride=2)
        self.leakyRelu3 = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(5*5*4*self.net_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyRelu1(x)
        x = self.conv2(x)
        x = self.leakyRelu2(x)
        x = self.conv3(x)
        x = self.leakyRelu3(x)
        x = x.reshape(-1, 5*5*4*self.net_dim)
        x = self.linear(x)
        return x


class ConvDiscriminatorFingervein1(nn.Module):
    def __init__(self):
        super(ConvDiscriminatorFingervein1, self).__init__()
        
        self.net_dim = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.net_dim, kernel_size=5, stride=2)
        self.leakyRelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=self.net_dim, out_channels=2*self.net_dim, kernel_size=5, stride=2)
        
        self.leakyRelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(in_channels=2*self.net_dim, out_channels=4 * self.net_dim, kernel_size=5, stride=2)
        
        self.leakyRelu3 = nn.LeakyReLU(0.2)

        
        self.linear = nn.Linear(5*13*4*self.net_dim, 1)

        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.leakyRelu1(x)
        

        x = self.conv2(x)
        
        x = self.leakyRelu2(x)
        

        x = self.conv3(x)
        
        x = self.leakyRelu3(x)
        

        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
    

    
    

    
    
    
    
    
    
    
    