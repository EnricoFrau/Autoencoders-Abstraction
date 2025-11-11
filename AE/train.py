import torch
import torch.nn as nn
import numpy as np
import random




def train(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    optimizer,
    writer,
    scheduler=None,
    save_tensorboard_parameters=False,
    starting_epoch = 0,
    l1_lambda = 0.0
):
    
    criterion = nn.MSELoss()

    global_batch_idx = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data) #+ l1_lambda * sum(p.abs().sum() for p in model.parameters())
            if l1_lambda:
                l1 = 0.0
                for p in model.parameters():
                    l1 += p.abs().sum()
                loss = loss + l1_lambda * l1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar(
            "Loss/train", train_loss / len(train_loader.dataset), global_step=(epoch + starting_epoch)
        )

        print(
            "Epoch: {}/{}, Average loss: {:.4f}".format(
                epoch+1, epochs, train_loss / len(train_loader.dataset)
            )
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()

        writer.add_scalar(
            "Loss/val", val_loss / len(val_loader.dataset), global_step=(epoch + starting_epoch)
        )

        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()
    print(
        f"Training completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )




def train_recursiveAE(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    optimizer,
    writer,
    scheduler=None,
    starting_epoch=0,
    latent_match_weight=1.0,
    detach_encoded_target=False,
    save_tensorboard_parameters=False,
    l1_lambda=0.0,
    recursive_lambda=0.1
):
    """
    Trains a model whose final output layer size is (input_dim + latent_dim).
    Loss = MSE(reconstruction) + latent_match_weight * MSE(latent_alignment).

    Assumes:
      model.encode(x) returns (B, latent_dim)
      model.decode(z) returns (B, input_dim + latent_dim) when configured for extended output
      model.input_dim and model.latent_dim are set
    """
    mse = nn.MSELoss()
    input_dim = model.input_dim
    latent_dim = model.latent_dim

    global_batch_idx = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_latent_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Encode once
            encoded_latent = model.encode(data)  # (B, latent_dim)
            encoded_target = encoded_latent.detach() if detach_encoded_target else encoded_latent

            # Decode from the latent to get extended output (input_dim + latent_dim)
            extended_out = model.decode(encoded_latent)  # (B, input_dim + latent_dim)
            decoded_part = extended_out[:, :input_dim]
            predicted_latent_part = extended_out[:, input_dim:]  # (B, latent_dim)

            # Match shapes for reconstruction loss
            if decoded_part.shape != data.shape:
                decoded_part = decoded_part.view_as(data)


            recon_loss = mse(decoded_part, data)
            latent_loss = mse(predicted_latent_part, encoded_target)
            loss = recon_loss + latent_match_weight * latent_loss

            if l1_lambda:
                l1 = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_latent_loss += latent_loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        n_train = len(train_loader.dataset)
        writer.add_scalar("Loss/train_total", total_train_loss / n_train, global_step=(epoch + starting_epoch))
        writer.add_scalar("Loss/train_recon", total_recon_loss / n_train, global_step=(epoch + starting_epoch))
        writer.add_scalar("Loss/train_latent", total_latent_loss / n_train, global_step=(epoch + starting_epoch))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Total: {(total_train_loss / n_train):.4f} | "
            f"Recon: {(total_recon_loss / n_train):.4f} | "
            f"Latent: {(total_latent_loss / n_train):.4f}"
        )

        # ---------------- Validation ----------------
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_latent = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device, non_blocking=True)

                encoded_latent = model.encode(data)
                extended_out = model.decode(encoded_latent)
                decoded_part = extended_out[:, :input_dim]
                predicted_latent_part = extended_out[:, input_dim:]

                if decoded_part.shape != data.shape:
                    decoded_part = decoded_part.view_as(data)

                recon_loss = mse(decoded_part, data)
                latent_loss = mse(predicted_latent_part, encoded_latent)
                loss = recon_loss + latent_match_weight * latent_loss

                val_total += loss.item()
                val_recon += recon_loss.item()
                val_latent += latent_loss.item()

        n_val = len(val_loader.dataset)
        writer.add_scalar("Loss/val_total", val_total / n_val, global_step=(epoch + starting_epoch))
        writer.add_scalar("Loss/val_recon", val_recon / n_val, global_step=(epoch + starting_epoch))
        writer.add_scalar("Loss/val_latent", val_latent / n_val, global_step=(epoch + starting_epoch))

        if save_tensorboard_parameters:
            for name, p in model.named_parameters():
                writer.add_histogram(name, p, global_step=(epoch + starting_epoch))
                if p.grad is not None:
                    writer.add_histogram(f"{name}.grad", p.grad, global_step=(epoch + starting_epoch))

    writer.close()
    print(
        f"Training completed. Final losses (per sample): "
        f"Train Total={total_train_loss / n_train:.4f}, Val Total={val_total / n_val:.4f}, "
        f"Train Recon={total_recon_loss / n_train:.4f}, Val Recon={val_recon / n_val:.4f}, "
        f"Train Latent={total_latent_loss / n_train:.4f}, Val Latent={val_latent / n_val:.4f}"
    )





# ------------------------------- Layer wise pretrain funcs ------------------------------

def layer_wise_pretrain_load_dict(ex_model, new_model):
    for i in range(len(ex_model.encoder)):
        if isinstance(ex_model.encoder[i], nn.Linear):
            # Get the corresponding layer in the new model
            for j in range(len(new_model.encoder)):
                if isinstance(new_model.encoder[j], nn.Linear):
                    if ex_model.encoder[i].weight.shape == new_model.encoder[j].weight.shape:
                        new_model.encoder[j].weight.data = ex_model.encoder[i].weight.data.clone()
                        new_model.encoder[j].bias.data = ex_model.encoder[i].bias.data.clone()
                        break

    # For the decoder (in reverse order since decoder is reversed)
    for i in range(len(ex_model.decoder)):
        if isinstance(ex_model.decoder[i], nn.Linear):
            # Get the corresponding layer in the new model
            for j in range(len(new_model.decoder)):
                if isinstance(new_model.decoder[j], nn.Linear):
                    if ex_model.decoder[i].weight.shape == new_model.decoder[j].weight.shape:
                        new_model.decoder[j].weight.data = ex_model.decoder[i].weight.data.clone()
                        new_model.decoder[j].bias.data = ex_model.decoder[i].bias.data.clone()
                        break



# -------------------------------- Training one feature at the time ----------------------------
 # neurons are 1-indexed


# ------------- Functions to mask parameters ------------------------

def mask_bottleneck_in_weight(weight, neuron_idx):      
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(weight)
        mask_add = torch.zeros_like(weight)
        mask_mul[active_neurons, :] = 1                 # weights have shape (latent_dim, encoder[-1])
        mask_add[train_neuron, :] = torch.randn((1, weight.size(1)))
        return weight * mask_mul + mask_add
    
def mask_bottleneck_in_bias(bias, neuron_idx):     
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(bias)
        mask_add = torch.zeros_like(bias)
        mask_mul[active_neurons] = 1
        mask_add[train_neuron] = torch.randn(1)
        return bias * mask_mul + mask_add
    
def mask_bottleneck_out_weight(weight, neuron_idx):    
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(weight)
        mask_add = torch.zeros_like(weight)
        mask_mul[:, active_neurons] = 1                 # weights have shape (latent_dim, encoder[-1])
        mask_add[:, train_neuron] = torch.randn((weight.size(0),))
        return weight * mask_mul + mask_add
    
def mask_bottleneck_out_bias(bias, neuron_idx):     
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(bias)
        mask_add = torch.zeros_like(bias)
        mask_mul[active_neurons] = 1
        mask_add[train_neuron] = torch.randn(1)
        return bias * mask_mul + mask_add


# ----------------- Functions for hook register -----------------

def mask_bottleneck_in_weight_grad(grad, neuron_idx, freeze_prev_neurons_train=True):     
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1, :] = 1                   # weights have shape (latent_dim, encoder[-1])
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons, :] = 1                 
    return grad * mask


def mask_bottleneck_in_bias_grad(grad, neuron_idx, freeze_prev_neurons_train=True):    
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1] = 1
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons] = 1
    return grad * mask


def mask_bottleneck_out_weight_grad(grad, neuron_idx, freeze_prev_neurons_train=True):    
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[:, neuron_idx-1] = 1                   # weights have shape (latent_dim, encoder[-1])
    else:
        active_neurons = list(range(neuron_idx))
        mask[:, active_neurons] = 1
    return grad * mask


def mask_bottleneck_out_bias_grad(grad, neuron_idx, freeze_prev_neurons_train=True):  
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1] = 1
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons] = 1
    return grad * mask



# -------------- TRAIN FUNCTIONS ----------------------



def train_single_neuron(
        model,
        epochs, 
        neuron_to_train, # 1-indexed
        train_loader, 
        val_loader,
        writer,
        lr = 1e-3,
        mask_weights = True,
        scheduler = None,
        freeze_prev_neurons_train = True,
        optimizer_func = torch.optim.Adam, 
        save_tensorboard_parameters = False,
        save_tensorboard_bottleneck_parameters = False,
        ):
    
    print(f"\n ------------ Training of neuron{neuron_to_train}---------------")

    # ----------------------------- Optimizer initialization -----------------------------

    optimizer = optimizer_func(model.parameters(), lr=lr, weight_decay=1e-5)

    # ----------------------------- Mask parameters and register hook -----------------------------------

    if mask_weights:
        with torch.no_grad():
            model.bottleneck_in[0].weight.copy_(
                mask_bottleneck_in_weight(model.bottleneck_in[0].weight, neuron_to_train)
                )
            model.bottleneck_in[0].bias.copy_(
                mask_bottleneck_in_bias(model.bottleneck_in[0].bias, neuron_to_train)
            )
            model.bottleneck_out[0].weight.copy_(
                mask_bottleneck_out_weight(model.bottleneck_out[0].weight, neuron_to_train)

                )
            model.bottleneck_out[0].bias.copy_(
                mask_bottleneck_out_bias(model.bottleneck_out[0].bias, neuron_to_train)
            )


    handle_weight_in = model.bottleneck_in[0].weight.register_hook(lambda grad: mask_bottleneck_in_weight_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_bias_in = model.bottleneck_in[0].bias.register_hook(lambda grad: mask_bottleneck_in_bias_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_weight_out = model.bottleneck_out[0].weight.register_hook(lambda grad: mask_bottleneck_out_weight_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_bias_out = model.bottleneck_out[0].bias.register_hook(lambda grad: mask_bottleneck_out_bias_grad(grad, neuron_to_train, freeze_prev_neurons_train))


    #-------------------------- Main loops --------------------------

    global_batch_idx = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # --------------------------- Weight optimization loop --------------------

        for data, _ in train_loader:
            data = data.to(model.device)    
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        print(
            "Epoch: {}/{}, Average loss: {:.4f}".format(
                (epoch+1), epochs, train_loss / len(train_loader.dataset)
            )
        )

        # -------------------------- Writer register scalars and parameters --------------------------

        writer.add_scalar(
            "Loss/train", train_loss / len(train_loader.dataset), global_step=(epoch + epochs * neuron_to_train) #to keep trace of the total number of epochs
        )

        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(model.device)
                output = model(data)
                loss = nn.MSELoss()(output, data)
                val_loss += loss.item()

        writer.add_scalar(
            "Loss/val", val_loss / len(val_loader.dataset), global_step=(epoch + epochs*neuron_to_train)
        )

        if save_tensorboard_bottleneck_parameters:
            for name, param in model.bottleneck_in.named_parameters():
                writer.add_histogram(name, param, global_step=(epoch + epochs*neuron_to_train))
                writer.add_histogram(f"{name}.grad", param.grad, global_step=(epoch + epochs*neuron_to_train))
            for name, param in model.bottleneck_out.named_parameters():
                writer.add_histogram(name, param, global_step=(epoch + epochs*neuron_to_train))
                writer.add_histogram(f"{name}.grad", param.grad, global_step=(epoch + epochs*neuron_to_train))


        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()

    # ------------------------ Remove hook and zero_grads ----------------------------

    handle_weight_in.remove()
    handle_bias_in.remove()
    handle_weight_out.remove()
    handle_bias_out.remove()

    optimizer.zero_grad()


    print(
        f"Training of neuron {neuron_to_train} completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )

    return 0



def train_ProgressiveAE(
        model,
        epochs_for_each_neuron,
        train_loader,
        val_loader,
        writer,
        lr = 1e-3,
        mask_weights = True,
        scheduler = None,
        freeze_prev_neurons_train = False,
        optimizer_func = torch.optim.Adam,
        save_tensorboard_parameters = False,
        save_tensorboard_bottleneck_parameters = False,
        ):
    
    print("\n-----------------------------TRAINING STARTED---------------------------- ")

    for neuron_idx in range(1, model.latent_dim + 1):     # neurons are 1-indexed

        train_single_neuron(
            model = model,
            epochs = epochs_for_each_neuron,
            neuron_to_train = neuron_idx,
            train_loader = train_loader,
            val_loader = val_loader,
            writer = writer,
            lr = lr,
            mask_weights = mask_weights,
            scheduler = scheduler,
            freeze_prev_neurons_train = freeze_prev_neurons_train,
            optimizer_func = optimizer_func,
            save_tensorboard_parameters = save_tensorboard_parameters,
            save_tensorboard_bottleneck_parameters = save_tensorboard_bottleneck_parameters,
            )


    print("------------------------------TRAINING ENDED-------------------------------")




# –––––––––––––––––––––––––––––––––– TRAIN WITH MIXED INPUTS –––––––––––––––––––––––––––––––––––––


def train_mixed_hidden(
    model,
    teacher_model,
    epochs,
    train_loader,
    val_loader,
    optimizer,
    writer,
    mix_percentage=0.1,
    scheduler=None,
    save_tensorboard_parameters=False,
    starting_epoch=0
):
    global_batch_idx = 0
    hidden_layers = model.number_of_hidden_layers

    teacher_model.eval()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(model.device)
            optimizer.zero_grad()

            if random.random() < mix_percentage:
                # Choose a random hidden layer before the bottleneck
                n = random.randint(1, hidden_layers)
                # Encode with teacher up to n-th hidden layer
                hidden = teacher_model.encode_truncated(data, num_hidden_layer=n)
                # Decode with model from that hidden layer
                output = model.decode_from_hidden(hidden, num_hidden_layer=n)
            else:
                # Standard forward pass
                output = model(data)

            if output.shape != data.shape:
                # Safe reshape when output is flat and target is image-shaped
                if output.dim() == 2 and data.dim() > 2 and output.size(0) == data.size(0):
                    output = output.view_as(data)


            loss = nn.MSELoss()(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar(
            "Loss/train", train_loss / len(train_loader.dataset), global_step=(epoch + starting_epoch)
        )

        print(
            "Epoch: {}/{}, Average loss: {:.4f}".format(
                epoch + 1, epochs, train_loss / len(train_loader.dataset)
            )
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(model.device)
                output = model(data)
                loss = nn.MSELoss()(output, data)
                val_loss += loss.item()

        writer.add_scalar(
            "Loss/val", val_loss / len(val_loader.dataset), global_step=(epoch + starting_epoch)
        )

        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()
    print(
        f"Training completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )











