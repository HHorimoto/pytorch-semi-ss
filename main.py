import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim

import yaml
import matplotlib.pyplot as plt

from src.data.dataset import create_dataset
from src.utils.seeds import fix_seed
from src.visualization.visualize import plot
from src.models.models import CNN
from src.models.coachs import Coach, CoachSSL

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    EPOCHS = config_file['config']['epochs']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']
    WEIGHT = config_file['config']['weight']
    METHOD = config_file['config']['method']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sup_loader, unsup_loader, test_loader = create_dataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    model = CNN(widen_factor=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_ce, loss_mse = nn.CrossEntropyLoss(), nn.MSELoss()

    if METHOD == 'ssl':
        coach = CoachSSL(model, sup_loader, unsup_loader, test_loader, loss_ce, loss_mse, 
                         WEIGHT, optimizer, device, EPOCHS)
    else: # METHOD == 'sl'
        coach = Coach(model, sup_loader, test_loader, loss_ce, optimizer, device, EPOCHS)
        
    coach.train_test()
    print("accuracy: ", coach.test_acc[-1])

    plot(coach.train_loss, coach.test_loss, 'loss')
    plot(coach.train_acc, coach.test_acc, 'accuracy')

if __name__ == "__main__":
    fix_seed()
    main()