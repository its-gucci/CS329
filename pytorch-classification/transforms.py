import torch

class AddGaussianNoise(object):
    
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, images):
        return images + sigma * torch.randn(images.size())