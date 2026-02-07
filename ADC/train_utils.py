import torch
import time
from torch import nn # For criterion
from tqdm import tqdm
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calibrate_model(model, calib_loader, device, portion=0.1):
    model.train()  # Enable train mode so observers collect stats
    print("Calibrating quantizers...")

    # Enable all relevant quantizers for calibration
    for _, module in model.named_modules():
        if hasattr(module, '_set_quantizer_state'):  # For LinearADC, LinearQuant
            module._set_quantizer_state(enabled=True)
        elif hasattr(module, 'enable'):  # Standalone quantizers
            module.enable()

    # Determine how many batches to use based on portion
    total_batches = len(calib_loader)
    num_batches = max(1, int(total_batches * portion))

    with torch.no_grad():  # No gradients needed
        for i, (inputs, _) in tqdm(
            enumerate(calib_loader), 
            total=num_batches, 
            desc="Calibrating", 
            leave=False
        ):
            inputs = inputs.to(device)
            model(inputs)  # Forward pass to update observers
            if i + 1 >= num_batches:
                break

    # After calibration, freeze quantizer parameters
    for _, module in model.named_modules():
        if hasattr(module, '_set_quantizer_state'):
            module._set_quantizer_state(enabled=False)
        elif hasattr(module, 'disable'):
            module.disable()

    print("Calibration done. Quantizer observers are now disabled.")
    model.eval()  # Return to eval mod



def calibrate_model3(model, calib_loader, device, portion=0.1):
    model.train()
    print("Calibrating quantizers...")

    print("Calibrating weight quantizers...")
    # Enable all relevant quantizers for calibration
    for mname, module in model.named_modules():
        #print(mname)
        if hasattr(module, '_set_quantizer_state'):  # For LinearADC, LinearQuant
            module.w_quantizer.update_state(0)
            module.w_quantizer(module.weight)
            module.w_quantizer.update_state(2)
            module.x_quantizer.update_state(0)
    
    print("Calibrating activation quantizers...")

    # Determine how many batches to use based on portion
    total_batches = len(calib_loader)
    num_batches = max(1, int(total_batches * portion))

    with torch.no_grad():  # No gradients needed
        for i, (inputs, _) in tqdm(
            enumerate(calib_loader), 
            total=num_batches, 
            desc="Calibrating", 
            leave=False
        ):
            inputs = inputs.to(device)
            model(inputs)  # Forward pass to update observers
            if i + 1 >= num_batches:
                break

    # After calibration, freeze quantizer parameters
    for _, module in model.named_modules():
        if hasattr(module, '_set_quantizer_state'):  # For LinearADC, LinearQuant
            module.w_quantizer.update_state(2)
            module.x_quantizer.update_state(2)
    model.eval()
    print("Calibration done. Quantizer observers are now disabled.")

def evaluate_model(model, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Calculating accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total, total_loss / total


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
        wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "test_loss" : epoch_test_loss, "test_acc" : epoch_test_acc})
        torch.save(model.state_dict(), model_name + "-st.pth")
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


import time

def finetune_model(model, optimizer, scheduler, train_loader, test_loader, criterion, device, num_epochs=20, model_name="Model", calib_loader=None, lambda_kurtosis=0.0):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    model.to(device)

    for mname, module in model.named_modules():
        # Set quantizer scales, zp to be learnt by gradient updates
        if hasattr(module, '_set_quantizer_state'):
            module.x_quantizer.update_state(1)
            module.w_quantizer.update_state(1)


    print(f"\n--- Finetuning {model_name} ---")
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
        wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "test_loss" : epoch_test_loss, "test_acc" : epoch_test_acc})
        torch.save(model.state_dict(), model_name + "-st.pth")
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


    