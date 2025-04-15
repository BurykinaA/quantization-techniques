import torchvision
import torchvision.transforms as transforms
import torch


def get_mnist_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    return train_loader, test_loader 