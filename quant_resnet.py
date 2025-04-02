import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model.resnet import resnet18

##############################################
# Custom config
##############################################
def get_qconfig_for_bitwidth(bitwidth):
    if bitwidth == 8:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    elif bitwidth == 6:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=63,  # 2**6 - 1
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-32,  # -2**(6-1)
                quant_max=31,   # 2**(6-1)-1
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    elif bitwidth == 4:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=15,  # 2**4 - 1
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-8,  # -2**(4-1)
                quant_max=7,   # 2**(4-1)-1
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    else:
        raise ValueError("Unsupported bitwidth")
    return torch.quantization.QConfig(activation=act_qconfig, weight=weight_qconfig)

##############################################
# experiments with MNIST on QAT for different bitnes (ResNet with num_classes=10)
##############################################
def perform_mnist_experiment(bitwidth, num_epochs=10):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = resnet18(pretrained=False, num_classes=10)
    modules_to_list = model.modules_to_fuse()
    
    model.eval()
    fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)
    fused_model.qconfig = get_qconfig_for_bitwidth(bitwidth)
    
    fused_model.train()
    torch.ao.quantization.prepare_qat(fused_model, inplace=True)
    
    optimizer = torch.optim.SGD(fused_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=f'./runs/mnist_qat_{bitwidth}bit')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = fused_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"MNIST {bitwidth}-bit - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.close()
    
    fused_model.apply(torch.ao.quantization.fake_quantize.disable_observer)
    quantized_model = torch.ao.quantization.convert(fused_model, inplace=True)
    
    quantized_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = quantized_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100.0 * correct / total
    print(f"MNIST {bitwidth}-bit - Test Accuracy: {accuracy:.2f}%")
    return accuracy

##############################################
# experiments with CIFAR on QAT for different bitnes (ResNet with num_classes=10)
##############################################
def perform_cifar_experiment(bitwidth, num_epochs=10):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model = resnet18(pretrained=False, num_classes=10)
    modules_to_list = model.modules_to_fuse()
    
    model.eval()
    fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)
    
    fused_model.qconfig = get_qconfig_for_bitwidth(bitwidth)
    
    fused_model.train()
    torch.ao.quantization.prepare_qat(fused_model, inplace=True)
    
    optimizer = torch.optim.SGD(fused_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=f'./runs/cifar_qat_{bitwidth}bit')
    
    fused_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = fused_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Bitwidth {bitwidth} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.close()
    
    fused_model.apply(torch.ao.quantization.fake_quantize.disable_observer)
    quantized_model = torch.ao.quantization.convert(fused_model, inplace=True)
    
    quantized_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = quantized_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Bitwidth {bitwidth} - CIFAR-10 Test Accuracy: {accuracy:.2f}%")
    return accuracy

##############################################
# run experiments; plot graphics
##############################################
if __name__ == '__main__':
    bitwidths = [8, 6, 4]
    mnist_accuracies = {}
    cifar_accuracies = {}
    
    for bw in bitwidths:
        print(f"\nЗапуск эксперимента MNIST для {bw}-битной квантизации")
        acc_mnist = perform_mnist_experiment(bitwidth=bw, num_epochs=10)
        mnist_accuracies[bw] = acc_mnist
        
        # print(f"\nЗапуск эксперимента CIFAR для {bw}-битной квантизации")
        # acc_cifar = perform_cifar_experiment(bitwidth=bw, num_epochs=10)
        # cifar_accuracies[bw] = acc_cifar
    
    plt.figure()
    plt.plot(list(mnist_accuracies.keys()), list(mnist_accuracies.values()), marker='o', label='MNIST')
    plt.xlabel('Битность')
    plt.ylabel('Точность на тесте (%)')
    plt.title('MNIST ResNet: Точность vs Квантизация (битность)')
    plt.grid(True)
    plt.legend()
    plt.savefig("mnist_accuracy_vs_bitwidth.png")
    plt.show()
    
    # plt.figure()
    # plt.plot(list(cifar_accuracies.keys()), list(cifar_accuracies.values()), marker='o', label='CIFAR')
    # plt.xlabel('Битность')
    # plt.ylabel('Точность на тесте (%)')
    # plt.title('CIFAR ResNet: Точность vs Квантизация (битность)')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("cifar_accuracy_vs_bitwidth.png")
    # plt.show()