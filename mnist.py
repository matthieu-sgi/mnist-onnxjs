import torch
import torch.nn as nn
import torchvision.transforms as t_transforms
import torch.optim.lr_scheduler as scheduler
import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter



# transforms_train = None
EPOCHS = 30

def custom_noise(tensor : torch.Tensor) -> torch.Tensor:
    return torch.clip(tensor + 0.1*torch.randn_like(tensor), 0, 1)

def custom_norm(tensor : torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5)

def custom_binarisation(tensor : torch.Tensor) -> torch.Tensor:
    
    return (tensor > 0.3).type(torch.float32)

transforms_train = t_transforms.Compose([
                                        # t_transforms.RandomAffine( degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=(-30, 30, -30, 30)),
                                        t_transforms.ToTensor(),
                                        custom_noise,

                                    #    t_transforms.Normalize((0.1307,), (0.3081,)),
                                    #    t_transforms.RandomRotation(10),
                                       t_transforms.RandomHorizontalFlip(p=0.5),
                                       t_transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                       t_transforms.RandomInvert(p=0.5),
                                       custom_binarisation,
                                        custom_norm,
                                       ])

transforms_test = t_transforms.Compose([ t_transforms.ToTensor(),
                                        custom_binarisation,
                                        custom_norm,
                                        
                                        # t_transforms.Normalize((0.1307,), (0.3081,)),
                                        ])

class MnistModelMLP(nn.Module):
    def __init__(self):
        super(MnistModelMLP, self).__init__()
        self.activation = nn.ReLU()
        self.linear_1 = nn.Linear(784, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 10)
        self.dropout_1 = nn.Dropout(0.2)


    def forward(self, x : torch.Tensor):
        x = x.view(-1, 784)
        output_1 = self.activation(self.linear_1(x))
        output_1 = self.dropout_1(output_1)
        output_2 = self.activation(self.linear_2(output_1))
        output_2 = self.dropout_1(output_2)
        output_3 = self.linear_3(output_2)
        return output_3
    
class MnistModelCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ELU()
        self.conv_block_1 = ConvBlock(in_channels=1, out_channels=16)
        # self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, groups=16) # MaxPool but with learning weigths
        self.ln1 = nn.LayerNorm([16, 14, 14])
        self.conv_block_2 = ConvBlock(in_channels=16, out_channels=32)
        self.max_pool_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, groups=32)
        self.ln2 = nn.LayerNorm([32, 7, 7])
        self.conv_block_3 = ConvBlock(in_channels=32, out_channels=64)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, groups=64)
        self.head = nn.Linear(64 * 1 *1, 10)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.max_pool_1(x)
        x = self.ln1(x)
        x = self.conv_block_2(x)
        x = self.max_pool_2(x)
        x = self.ln2(x)
        x = self.conv_block_3(x)
        x = self.final_conv(x)
        x = self.activation(x)
        x = x.view(-1, 64*1*1)
        x = self.head(x)
        # x = self.softmax(x)
        return x




        

class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels :int, *, kernel_size : int = 3) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels = in_channels,
                                out_channels= out_channels,
                                kernel_size=1)
        
        self.dw_conv = nn.Conv2d(in_channels = out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 groups=out_channels,
                                 padding=kernel_size//2)
        
        self.conv_2 = nn.Conv2d(in_channels = out_channels,
                                out_channels=out_channels,
                                kernel_size=1)
        
        self.activation = nn.ELU()
        self.input_channels = in_channels
        self.output_channels = out_channels


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # input_ = x
        # input_ = input_.repeat(1, self.output_channels//self.input_channels, 1, 1)
        x = self.conv_1(x)
        # x = self.activation(x)
        x = self.dw_conv(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        # x = input_ + x
        return x
    

    


def train(
        model : nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion,
        train_loader : DataLoader,
        lr_scheduler : scheduler.OneCycleLR 
        ) -> None:
    
    model.train()

    total_loss = 0
    total_accuracy = 0
    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc='Training')):

        # Put the optimizer to zero
        optimizer.zero_grad()

        # Forward pass
        output : torch.Tensor = model(data)

        loss = criterion(output, target)

        # Backward pass (backpropagation)
        loss.backward()
        
        # Update the weights
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        total_accuracy += (target == output.argmax(dim=1)).sum().item()

    total_accuracy /= len(train_loader.dataset)
    total_loss /= len(train_loader)
    total_accuracy *= 100
    print(f'1 training epoch ended, Total loss: {total_loss}, Total accuracy: {total_accuracy}')
    return total_loss, total_accuracy

def validate(
        model : nn.Module,
        criterion,
        test_loader : DataLoader
        ) -> None:

    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, desc='Validation'):

            output : torch.Tensor = model(data)

            test_loss += criterion(output, target).item()

            accuracy = (target == torch.argmax(input=output, dim=1)).sum().item()

            test_accuracy += accuracy  

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    test_accuracy *= 100

    print(f'Validating : Average loss: {test_loss}, Accuracy: {test_accuracy}\n')
    return test_loss, test_accuracy


def epochs(
        model : nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion : nn.Module,
        train_loader : torch.utils.data.DataLoader,
        test_loader : torch.utils.data.DataLoader,
        epochs_ : int,
        lr_scheduler : scheduler.OneCycleLR
        ) -> None:

    writer = SummaryWriter()
    writer.add_graph(model, torch.rand(1, 1, 28, 28))
    best_loss = 99999
    for epoch in range(1, epochs_ + 1):
        train_loss, train_accuracy = train(model, optimizer, criterion, train_loader, lr_scheduler)
        test_loss, test_accuracy = validate(model, criterion, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print('Model saved')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('Learning rate', lr_scheduler.get_last_lr()[0], epoch)
    writer.close()

if __name__ == '__main__':

    training_dataset = MNIST(root = './datasets',train=True, download=True, transform=transforms_train)
    test_dataset = MNIST(root = './datasets',train=False, download=True, transform=transforms_test)

    train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)

    model = MnistModelCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learning_rate_scheduler = scheduler.OneCycleLR(optimizer=optimizer,
                                                   max_lr=1e-2,
                                                   epochs=EPOCHS,
                                                   steps_per_epoch=len(train_dataloader))

    epochs(model,
           optimizer,
           criterion,
           train_dataloader,
           test_dataloader,
           EPOCHS,
           lr_scheduler= learning_rate_scheduler)


