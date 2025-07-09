import torch
import time
from torch import nn # For criterion
from tqdm import tqdm

def calibrate_model(model, calib_loader, device):
    model.train() # Set to train mode for observers to work
    print("Calibrating quantizers...")
    # Ensure all relevant quantizers are enabled for calibration
    for _, module in model.named_modules():
        if hasattr(module, '_set_quantizer_state'): # For LinearADC, LinearQuant
            module._set_quantizer_state(enabled=True)
        elif hasattr(module, 'enable'): # For standalone quantizers if any (not typical in layers)
             module.enable()


    with torch.no_grad(): # No gradients needed for calibration
        for i, (inputs, _) in enumerate(calib_loader):
            inputs = inputs.to(device)
            model(inputs)
            #model(inputs.view(inputs.size(0), -1)) # Forward pass to update observers
            if i >= 20:  # Calibrate on a few batches (e.g., 20 batches)
                break
    
    # After calibration, disable observers so their parameters (scale/zp) are fixed
    for _, module in model.named_modules():
        if hasattr(module, '_set_quantizer_state'):
             module._set_quantizer_state(enabled=False) # Observers no longer update scale/zp
        elif hasattr(module, 'disable'):
             module.disable()

    print("Calibration done. Quantizer observers are now disabled.")
    model.eval() # Set back to eval mode or train() will be called by the main loop again


def train_model(model, optimizer, scheduler, train_loader, test_loader, criterion, device, num_epochs=20, model_name="Model", calib_loader=None, lambda_kurtosis=0.0):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    model.to(device)

    # Perform calibration before the first epoch if a calibration loader is provided
    if calib_loader:
        calibrate_model(model, calib_loader, device)
    else: # If no specific calib_loader, assume quantizers are part of layers and handle enabling/disabling there.
          # Or, could do a brief calibration on a few batches of train_loader here.
          # For this setup, LinearADC/LinearQuant enable observers in train() and disable in eval().
          # A dedicated calibration step before training loop is cleaner.
          # Let's try to calibrate on first few train batches if no calib_loader.
        print("No dedicated calibration loader. Calibrating on initial training batches...")
        calibrate_model(model, train_loader, device)


    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train() # Set model to training mode. This will enable observers in our custom layers.
        
        # Important: After calibration, observers in custom layers should ideally be fixed (disabled).
        # The train() call above will re-enable them if _set_quantizer_state(True) is used.
        # We need a way to distinguish between "training the model weights" vs "calibrating quantizers".
        # The `calibrate_model` function now handles disabling observers AFTER calibration.
        # So, the `model.train()` here is for training model parameters, not re-calibrating.
        # The quantizers inside LinearADC/LinearQuant will have `enabled=False` due to `calibrate_model`.
        # This is correct: scale/zp are fixed, STE passes gradients.

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Add kurtosis penalty if lambda_kurtosis is set
            if lambda_kurtosis > 0:
                kurtosis_penalty = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.dim() > 1: # Apply to weight tensors (e.g., fc.weight)
                        W = param
                        mu_W = torch.mean(W)
                        sigma_W = torch.std(W)
                        if sigma_W > 1e-5: # Avoid division by zero or very small std
                            kappa_l = torch.mean(torch.pow((W - mu_W) / sigma_W, 4))
                            kurtosis_penalty += kappa_l
                loss = criterion(outputs, labels)
                loss += lambda_kurtosis * kurtosis_penalty
            else:
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
        scheduler.step()

        model.eval() # Set model to evaluation mode. This disables observers in custom layers.
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Validation epoch {epoch+1} / {num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                #inputs = inputs.view(inputs.size(0), -1)
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
    if test_accuracies:
        print(f"Final Test Accuracy for {model_name}: {test_accuracies[-1]:.2f}%")
    else:
        print(f"No test accuracies recorded for {model_name}.")
    return train_losses, train_accuracies, test_losses, test_accuracies
