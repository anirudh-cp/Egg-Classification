# Model and data manipulation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Display functions
import matplotlib.pyplot as plt

# Set seeds
import time
import random

# Save data
from datetime import datetime
import json


DATA_PATH = 'Data'
HELPER_DISPLAY_ROWS = 2
HELPER_DISPLAY_COLS = 2
RESIZED = (140, 250)
CLASSES = ('A', 'B')

EPOCHS = 10
BATCH_SIZE = 32
LEARN_RT = 0.001
PATIENCE = 5

OPTIMIZER = torch.optim.Adam
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau


def set_device():
    # Set device
    cuda = True
    device = torch.device("cuda" if (
        torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print(device)


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def imshow(img, labels, classes):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()

    fig, ax = plt.subplots(HELPER_DISPLAY_ROWS, HELPER_DISPLAY_COLS)
    for index, image in enumerate(npimg):
        print(image.shape)
        axes = ax[index//HELPER_DISPLAY_ROWS, index%HELPER_DISPLAY_COLS]
        axes.imshow(np.transpose(image, (1, 2, 0)))
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_title(classes[labels[index]])

    fig.tight_layout()
    plt.show()

    
class ImageClassificationBase(nn.Module):
    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = self.accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# Edge detection then scale, consitency, centerNet objects as points
class CNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # VGG model
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), # output: 64 x 70 x 125

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2, padding=(0, 1)), # output: 256 x 35 x 63

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2, padding=(1, 1)), # output: 256 x 18 x 32

            nn.Flatten(), 
            nn.Linear(256*18*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2))
        
        
    def forward(self, xb):
        return self.network(xb)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(model, train_loader, val_loader, optimizer, scheduler, patienceInit):
    history = []
    best_val_loss = np.inf
    patience = patienceInit
    
    for epoch in range(EPOCHS):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            
        # Validation phase
        result = evaluate(model, val_loader)
        
        # Change learning rate if required
        scheduler.step(result['val_loss'])
        
        # Early stopping
        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            best_model = model
            patience = patienceInit  # reset _patience
        else:
            patience -= 1
        
        # Logging
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['epoch'] = epoch
        result['lr'] = optimizer.param_groups[0]['lr']
        result['patience'] = patience
        
        # model.epoch_end(epoch, result)
        history.append(result)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {result['train_loss']:.5f}, "
            f"val_loss: {result['val_loss']:.5f}, "
            f"val_acc: { result['val_acc']:.5f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.2E}, "
            f"patience: {patience}"
        )
        
        if not patience:  # 0
            print("Stopping early!")
            break
    
    return history


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(f'Analysis/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Accuracy_vs_Epoch.png')
    # plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(f'Analysis/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_Loss_vs_Epoch.png')
    # plt.show()
    

def main():
    set_seeds(int(time.time()))
    set_device()

    TrainData = datasets.ImageFolder(DATA_PATH + '/train', transform=transforms.Compose([transforms.Resize(RESIZED), transforms.ToTensor()]))
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, shuffle=True, batch_size=BATCH_SIZE)
    
    TestData = datasets.ImageFolder(DATA_PATH + '/test', transform=transforms.Compose([transforms.Resize(RESIZED), transforms.ToTensor()]))
    TestDataLoader = torch.utils.data.DataLoader(TestData, shuffle=True, batch_size=BATCH_SIZE)
    
    model = CNN()
    
    optimizer = OPTIMIZER(model.parameters(), lr=LEARN_RT)
    scheduler = SCHEDULER(optimizer, mode="min", factor=0.1, patience=3)
    
    history = fit(model, TrainDataLoader, TestDataLoader, optimizer, scheduler, PATIENCE)
    
    torch.save(model.state_dict(), f'Models/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')

    plot_accuracies(history)
    plot_losses(history)
    
    with open(f'Analysis/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_History.json', 'w', encoding ='utf8') as json_file:
        json.dump(history, json_file, ensure_ascii = True, indent=4)

    '''
    TestDataIter = iter(TestDataLoader)
    images, labels = TestDataIter.next()
    
    # show images
    imshow(images, labels, CLASSES)
    '''


if __name__ == '__main__':
    main()
