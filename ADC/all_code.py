import torch

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: just pass gradient through
        return grad_output

def ste_floor(x):
    return FloorSTE.apply(x)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: just pass gradient through
        return grad_output

def ste_round(x):
    return RoundSTE.apply(x)


import torch
from torch.ao.quantization.observer import MinMaxObserver
from torch import nn


class AffineQuantizerPerTensor(nn.Module):
    def __init__(self, bx=8):
        super().__init__()
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=2**bx - 1)
        self.scale = None
        self.zero_point = None
        self.bx = bx
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            self.observer(x)
            self.scale, self.zero_point = self.observer.calculate_qparams()

        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Quantizer must be calibrated before use.")

        scale = self.scale.to(x.device)
        zero_point = self.zero_point.to(scale.dtype).to(x.device)

        xq = ste_round(x / scale + zero_point)
        xq = torch.clamp(xq, self.observer.quant_min, self.observer.quant_max)
        return xq


class SymmetricQuantizerPerTensor(nn.Module):
    def __init__(self, bw=8):
        super().__init__()
        maxval = 2 ** (bw - 1) - 1
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, quant_min=-maxval, quant_max=maxval)
        self.scale = None
        self.bw = bw
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            self.observer(x)
            self.scale, self.zero_point = self.observer.calculate_qparams()

        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Quantizer must be calibrated before use.")

        scale = self.scale.to(x.device)

        xq = ste_round(x / scale)
        xq = torch.clamp(xq, self.observer.quant_min, self.observer.quant_max)
        return xq

class ADCQuantizer(nn.Module):
    def __init__(self, M, bx, bw, ba = 8, k = 4):
        super().__init__()
        self.delta = 2 * M * (2 ** bx - 1) * (2 ** (bw - 1) - 1) / ((2 ** ba - 1) * k)
        self.M = M
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = ste_floor(x / self.delta)
        mnval = -2 ** (self.ba - 1)
        mxval = 2 ** (self.ba - 1) - 1
        xq = torch.clamp(xq, mnval, mxval)
        return xq


class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)

    def dequantize(self, yq):

        # yq [B, O]

        # y = sum xq_i * wq_i
        # yq = y /

        y = yq * self.adc_quantizer.delta
        # The dequantization formula for ADC involves more than just scaling.
        # The original forward pass is:
        # xq = round(x/scale_x + zp_x)
        # wq = round(w/scale_w)
        # y_adc_quantized = floor( (sum(xq * wq)) / delta_adc )
        #
        # To dequantize y_adc_quantized back to an approximation of sum( (x/scale_x + zp_x) * (w/scale_w) ):
        # y_approx_scaled_sum = y_adc_quantized * delta_adc
        #
        # This y_approx_scaled_sum is an approximation of sum(xq * wq).
        # To get back to the original unquantized output scale, we need to consider how xq and wq were derived.
        # sum(xq * wq) approximates sum( (x/scale_x + zp_x) * (w/scale_w) )
        # = sum( x*w / (scale_x*scale_w) + zp_x*w / scale_w )
        #
        # The dequantization here aims to reverse the ADC quantization and the effect of input/weight quantizers.
        # This part seems complex and might need careful derivation based on the exact ADC formulation.
        # The provided dequantize function:
        # out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        # out = out * self.x_quantizer.scale * self.w_quantizer.scale
        # This attempts to undo the scaling and zero-point effects.
        # Let's assume this dequantization is correct as per the ADC paper/logic.
        # The term self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        # seems to try to correct for the zero_point in activations.
        # And then the final scaling by self.x_quantizer.scale * self.w_quantizer.scale reverses the input and weight scaling.

        # y_dequant_adc = yq * self.adc_quantizer.delta # This is sum(xq*wq) approximately
        # Now, we need to relate sum(xq*wq) back to the original y = x*w + b
        # xq = x/scale_x + zp_x  => x = (xq - zp_x)*scale_x
        # wq = w/scale_w         => w = wq*scale_w
        # y = sum ( (xq_i - zp_x) * scale_x * wq_i * scale_w )
        # y = scale_x * scale_w * sum ( (xq_i - zp_x) * wq_i )
        # y = scale_x * scale_w * ( sum(xq_i * wq_i) - zp_x * sum(wq_i) )
        #
        # So, if y_dequant_adc is sum(xq_i * wq_i):
        # out = self.x_quantizer.scale * self.w_quantizer.scale * \
        #       (y_dequant_adc - self.x_quantizer.zero_point * self.w_quantizer(self.weight).sum(axis=-1))

        # The current implementation is:
        # y = yq * self.adc_quantizer.delta
        # out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        # out = out * self.x_quantizer.scale * self.w_quantizer.scale
        #
        # Let's re-evaluate self.weight.sum(axis=-1). This sums weights for each output neuron.
        # wq = self.w_quantizer(self.weight)
        # The term zp_x * sum(wq_i) requires the quantized weights wq.
        # The term self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        # is an approximation of zp_x * sum(wq_i) if self.weight.sum(axis=-1) / self.w_quantizer.scale approximates sum(wq_i).

        y_sum_xq_wq = yq * self.adc_quantizer.delta
        wq = self.w_quantizer(self.weight) # Get quantized weights

        # We need to compute sum_i ( (xq_i - zp_x) * wq_i ) * scale_x * scale_w
        # This is ( sum_i(xq_i * wq_i) - zp_x * sum_i(wq_i) ) * scale_x * scale_w
        # sum_i(xq_i * wq_i) is approximated by y_sum_xq_wq
        # sum_i(wq_i) is wq.sum(dim=1) for each output neuron (or dim=0 if weights are [out, in])
        # Weight shape is [out_features, in_features]
        sum_wq_per_output = wq.sum(dim=1, keepdim=True).T # Ensure broadcasting [1, out_features]

        # zero_point is scalar for per-tensor.
        # The original y = xw + b, so x is [B, in_features], w is [out_features, in_features]
        # y_unquant = F.linear(x, w)
        # If we use xq and wq:
        # x_dequant = (xq - self.x_quantizer.zero_point) * self.x_quantizer.scale
        # w_dequant = wq * self.w_quantizer.scale
        # y_dequant_output = F.linear(x_dequant, w_dequant)
        # y_dequant_output = F.linear((xq - self.x_quantizer.zero_point) * self.x_quantizer.scale, wq * self.w_quantizer.scale)
        # y_dequant_output = self.x_quantizer.scale * self.w_quantizer.scale * F.linear(xq - self.x_quantizer.zero_point, wq)
        # y_dequant_output = self.x_quantizer.scale * self.w_quantizer.scale * (F.linear(xq, wq) - self.x_quantizer.zero_point * wq.sum(dim=1))
        #
        # F.linear(xq, wq) is what y_sum_xq_wq represents.
        # So, y_dequant_output = self.x_quantizer.scale * self.w_quantizer.scale * (y_sum_xq_wq - self.x_quantizer.zero_point * sum_wq_per_output)
        
        # Make sure zero_point has the correct device and dtype
        zp_x = self.x_quantizer.zero_point.to(y_sum_xq_wq.device, dtype=y_sum_xq_wq.dtype)

        out = self.x_quantizer.scale.to(y_sum_xq_wq.device) * \
              self.w_quantizer.scale.to(y_sum_xq_wq.device) * \
              (y_sum_xq_wq - zp_x * sum_wq_per_output)
        return out

    def train(self, mode=True):
        super().train(mode)
        if (mode == True):
            self.x_quantizer.enable()
            self.w_quantizer.enable()
        else:
            self.x_quantizer.disable()
            self.w_quantizer.disable()
        return self

    def eval(self, mode=True):
        super().eval(mode)
        self.train(not mode)
        return self

    def forward(self, x):
        xq = self.x_quantizer(x)
        wq = self.w_quantizer(self.weight)
        # y = nn.functional.linear(xq, wq) # This is sum(xq_i * wq_i)
        # For ADC, the input to adc_quantizer should be the sum before any dequantization
        y_for_adc = nn.functional.linear(xq, wq) # This is effectively sum(xq_i * wq_i) for each output neuron
        yq_adc = self.adc_quantizer(y_for_adc)
        out = self.dequantize(yq_adc)
        if self.bias is not None:
            out = out + self.bias
        return out
    

   # Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), # Input layer (28x28 images flattened)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # Output layer (10 classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        return self.layers(x) 



# Define the MLP model
class MLPADC(nn.Module):
    def __init__(self, bx=8, bw=8, ba=8, k=4,):
        super(MLPADC, self).__init__()
        self.layers = nn.Sequential(
            LinearADC(784, 256, bx, bw, ba, k), # Input layer (28x28 images flattened)
            nn.ReLU(),
            LinearADC(256, 128, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(128, 10, bx, bw, ba, k) # Output layer (10 classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        return self.layers(x)
    

from torchvision import datasets, transforms

# ... (rest of your code)

# Define a transform to normalize the data
transform_fm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)) # Normalize using mean and std dev of MNIST
])

# Download and load the training data
train_dataset_fm = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_fm)

# Download and load the test data
test_dataset_fm = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_fm)


model_adc4 = MLPADC(bx=4, bw=4, ba=8, k=4).to(device)
optimizer = torch.optim.Adam(model_adc4.parameters(), lr=0.001)
train_losses_adc4, train_accuracies_adc4, test_losses_adc4, test_accuracies_adc4 = train(model_adc4, optimizer, train_loader, test_loader, num_epochs=20)

class LinearQuant(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, bias=True):
        super(LinearQuant, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.x_quantizer.enable()
            self.w_quantizer.enable()
        else:
            self.x_quantizer.disable()
            self.w_quantizer.disable()
        return self

    def eval(self): # PyTorch standard is eval() without mode
        super().eval()
        self.x_quantizer.disable()
        self.w_quantizer.disable()
        return self

    def forward(self, x):
        # Quantize activations and weights
        xq = self.x_quantizer(x)
        wq = self.w_quantizer(self.weight)

        # Dequantize for linear operation (simulating quantized hardware)
        # x_dequant = (xq - self.x_quantizer.zero_point) * self.x_quantizer.scale
        # w_dequant = wq * self.w_quantizer.scale

        # For affine x: x_dequant = (xq - zp_x) * scale_x
        # For symmetric w: w_dequant = wq * scale_w
        # Ensure scales and zero_points are on the correct device and dtype
        
        scale_x = self.x_quantizer.scale.to(xq.device)
        zp_x = self.x_quantizer.zero_point.to(xq.device, dtype=xq.dtype) # Match xq's dtype for subtraction
        
        x_dequant = (xq.to(scale_x.dtype) - zp_x) * scale_x # Cast xq to float before op if needed

        scale_w = self.w_quantizer.scale.to(wq.device)
        w_dequant = wq.to(scale_w.dtype) * scale_w # Cast wq to float before op if needed


        # Perform linear operation with dequantized values
        out = nn.functional.linear(x_dequant, w_dequant, self.bias)
        return out


   # Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), # Input layer (28x28 images flattened)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # Output layer (10 classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        return self.layers(x) 



# Define the MLP model
class MLPADC(nn.Module):
    def __init__(self, bx=8, bw=8, ba=8, k=4,):
        super(MLPADC, self).__init__()
        self.layers = nn.Sequential(
            LinearADC(784, 256, bx, bw, ba, k), # Input layer (28x28 images flattened)
            nn.ReLU(),
            LinearADC(256, 128, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(128, 10, bx, bw, ba, k) # Output layer (10 classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        return self.layers(x)
    

class MLPQuant(nn.Module):
    def __init__(self, bx=8, bw=8):
        super(MLPQuant, self).__init__()
        self.layers = nn.Sequential(
            LinearQuant(784, 256, bx, bw),
            nn.ReLU(),
            LinearQuant(256, 128, bx, bw),
            nn.ReLU(),
            LinearQuant(128, 10, bx, bw)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        return self.layers(x)


from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time # For logging training time

# Define a transform to normalize the data
transform_fm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)) # Normalize using mean and std dev of MNIST
])

# Download and load the training data
train_dataset_fm = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_fm)

# Download and load the test data
test_dataset_fm = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_fm)

# Data loaders
train_loader = DataLoader(train_dataset_fm, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset_fm, batch_size=1000, shuffle=False)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training function (assuming this is defined elsewhere or needs to be added)
# For now, I will create a basic training loop here.
# If you have a specific 'train' function, we can integrate that.

def train_model(model, optimizer, train_loader, test_loader, criterion, num_epochs=20, model_name="Model"):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    model.to(device)

    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Calibration phase for quantizers in the first part of the first epoch
        if epoch == 0:
            print("Calibrating quantizers...")
            for i, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                model(inputs.view(inputs.size(0), -1))
                if i > 20: # Calibrate on a few batches
                    break 
            print("Calibration done.")
            # After calibration, ensure quantizers are not re-calculating scale/zp
            for _, module in model.named_modules():
                if hasattr(module, 'x_quantizer') and hasattr(module.x_quantizer, 'disable'): # for LinearQuant/LinearADC
                    module.x_quantizer.disable() # Disable further observer updates
                if hasattr(module, 'w_quantizer') and hasattr(module.w_quantizer, 'disable'): # for LinearQuant/LinearADC
                    module.w_quantizer.disable()


        model.train() # Ensure model is in training mode after calibration run
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1) # Flatten

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(inputs.size(0), -1) # Flatten
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = 100. * correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} => "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

    end_time = time.time()
    print(f"Training {model_name} finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Test Accuracy for {model_name}: {test_accuracies[-1]:.2f}%")
    return train_losses, train_accuracies, test_losses, test_accuracies

# --- Experiment Setup ---
num_epochs_exp = 5 # Using fewer epochs for a quick comparison, adjust as needed
criterion = nn.CrossEntropyLoss()

# Parameters for quantization
bx_val = 4
bw_val = 4
ba_val = 8 # For ADC
k_val = 4  # For ADC

# 1. Train and evaluate MLPADC
print("\n--- Experiment with MLPADC ---")
model_adc = MLPADC(bx=bx_val, bw=bw_val, ba=ba_val, k=k_val)
optimizer_adc = optim.Adam(model_adc.parameters(), lr=0.001)
train_losses_adc, train_accs_adc, test_losses_adc, test_accs_adc = train_model(
    model_adc, optimizer_adc, train_loader, test_loader, criterion, num_epochs=num_epochs_exp, model_name=f"MLPADC (bx={bx_val}, bw={bw_val}, ba={ba_val})"
)

# 2. Train and evaluate MLPQuant (standard quantization)
print("\n--- Experiment with MLPQuant (Standard Quantization) ---")
model_quant = MLPQuant(bx=bx_val, bw=bw_val)
optimizer_quant = optim.Adam(model_quant.parameters(), lr=0.001)
train_losses_quant, train_accs_quant, test_losses_quant, test_accs_quant = train_model(
    model_quant, optimizer_quant, train_loader, test_loader, criterion, num_epochs=num_epochs_exp, model_name=f"MLPQuant (bx={bx_val}, bw={bw_val})"
)


# --- Results Logging ---
print("\n\n--- Experiment Results ---")
print(f"Parameters: bx={bx_val}, bw={bw_val}, ba={ba_val} (for ADC), k={k_val} (for ADC), Epochs={num_epochs_exp}")

print("\nMLPADC Performance:")
print(f"  Final Train Accuracy: {train_accs_adc[-1]:.2f}%")
print(f"  Final Test Accuracy:  {test_accs_adc[-1]:.2f}%")
print(f"  Final Train Loss:     {train_losses_adc[-1]:.4f}")
print(f"  Final Test Loss:      {test_losses_adc[-1]:.4f}")

print("\nMLPQuant (Standard Quantization) Performance:")
print(f"  Final Train Accuracy: {train_accs_quant[-1]:.2f}%")
print(f"  Final Test Accuracy:  {test_accs_quant[-1]:.2f}%")
print(f"  Final Train Loss:     {train_losses_quant[-1]:.4f}")
print(f"  Final Test Loss:      {test_losses_quant[-1]:.4f}")


# The line you had:
# model_adc4 = MLPADC(bx=4, bw=4, ba=8, k=4).to(device)
# optimizer = torch.optim.Adam(model_adc4.parameters(), lr=0.001)
# train_losses_adc4, train_accuracies_adc4, test_losses_adc4, test_accuracies_adc4 = train(model_adc4, optimizer, train_loader, test_loader, num_epochs=20)
# This can be replaced by the experiment block above.
# If you want to keep your original training call, ensure the 'train' function is defined and handles calibration correctly.