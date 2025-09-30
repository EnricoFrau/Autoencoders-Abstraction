import torch
from torch import nn, optim


class AE_0(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        device,
        hidden_layers=1,
        decrease_rate=0.5,
        activation_fn=nn.ReLU,
        output_activation_encoder=nn.Sigmoid,
        output_activation_decoder=None,
        BatchNorm=False,
        LayerNorm=False,
        he_init=False,
        set_bias=None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.number_of_hidden_layers = hidden_layers
        self.device = device
        self.activation_fn = activation_fn

        encoder_neurons_sizes = [input_dim]

        for _ in range(self.number_of_hidden_layers):
            encoder_neurons_sizes.append(int(encoder_neurons_sizes[-1] * decrease_rate))

        encoder_neurons_sizes.append(latent_dim)

        encoder_layers = []
        for i in range(len(encoder_neurons_sizes) - 1):
            encoder_layers.append(
                nn.Linear(encoder_neurons_sizes[i], encoder_neurons_sizes[i + 1])
            )
            if i < len(encoder_neurons_sizes) - 2:
                if BatchNorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_neurons_sizes[i + 1]))
                if LayerNorm:
                    encoder_layers.append(nn.LayerNorm(encoder_neurons_sizes[i + 1]))
                encoder_layers.append(self.activation_fn())
            elif output_activation_encoder is not None:
                encoder_layers.append(output_activation_encoder())

        self.encoder = nn.Sequential(*encoder_layers).to(
            self.device
        )  # TOCHECK TO DEVICE

        decoder_neurons_sizes = list(reversed(encoder_neurons_sizes))
        decoder_layers = []
        for i in range(len(decoder_neurons_sizes) - 1):
            decoder_layers.append(
                nn.Linear(decoder_neurons_sizes[i], decoder_neurons_sizes[i + 1])
            )
            if i < len(decoder_neurons_sizes) - 2:
                if BatchNorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_neurons_sizes[i + 1]))
                if LayerNorm:
                    decoder_layers.append(nn.LayerNorm(decoder_neurons_sizes[i + 1]))
                decoder_layers.append(self.activation_fn())
            elif output_activation_decoder is not None:
                decoder_layers.append(output_activation_decoder())

        self.decoder = nn.Sequential(*decoder_layers).to(
            self.device
        )  # TOCHECK TO DEVICE


        if set_bias is not None:
            self.set_bias(set_bias)

        if he_init:
            self.he_initialization



    def encode_truncated(self, data, num_hidden_layer):
        """
        Encodes the data up to the specified hidden layer (not including the bottleneck).
        num_hidden_layer: index of the hidden layer (starting from 1)
        """
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)
        else:
            data_flat = data

        x = data_flat
        layer_count = 0
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                layer_count += 1
                if layer_count == num_hidden_layer:
                    break
        return x



    def decode_from_hidden(self, data, num_hidden_layer):
        """
        Decodes data starting from the output of a given encoder hidden layer.
        num_hidden_layer: index of the encoder hidden layer (starting from 1)
        """

        hidden = self.encode_truncated(data, num_hidden_layer)

        decoded = self.decoder(hidden)
        return decoded



    def encode(self, data):
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)  # data_flat has size (batch_size(64), 28, 28)
        else:
            data_flat = data
        return self.encoder(data_flat)
    


    def decode(self, data):
        return self.decoder(data)



    def forward(self, data): # batch has size (batch_size(64), 28, 28)
        original_shape = data.shape
        decoded = self.decode(self.encode(data))
        # Reshape decoded output back to original input shape
        if original_shape != decoded.shape:
            decoded = decoded.view(original_shape)
        return decoded



    def set_bias(self, value):
        with torch.no_grad():
            for i in range((len(self.encoder))):
                if isinstance(self.encoder[i], nn.Linear):
                    nn.init.constant_(self.encoder[i].bias, value)
            for i in range((len(self.decoder))):
                if isinstance(self.decoder[i], nn.Linear):
                    nn.init.constant_(self.decoder[i].bias, value)

    def he_initialization(self):
        with torch.no_grad():
            for i in range((len(self.encoder))):
                if isinstance(self.encoder[i], nn.Linear):
                    nn.init.kaiming_normal_(self.encoder[i].weight)
            for i in range((len(self.decoder))):
                if isinstance(self.decoder[i], nn.Linear):
                    nn.init.kaiming_normal_(self.decoder[i].weight)


    



class ProgressiveAE(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            device,
            num_hidden_layers = 1,
            decrease_rate = 0.5,
            activation_fn = nn.ReLU,
            bottleneck_in_fn = nn.Sigmoid
        ):
        
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.activation_fn = activation_fn
        self.bottleneck_in_fn = bottleneck_in_fn


        # --------------Encoder----------------

        encoder_layers_sizes = [input_dim]
        for i in range(num_hidden_layers):
            encoder_layers_sizes.append(int(encoder_layers_sizes[-1] * decrease_rate))

        encoder_layers = []                             # creates the encoder without the bottleneck
        for i in range(len(encoder_layers_sizes) - 1):
            encoder_layers.append(
                nn.Linear(encoder_layers_sizes[i], encoder_layers_sizes[i + 1])
                )
            encoder_layers.append(
                activation_fn()
                )
        self.amputated_encoder = nn.Sequential(*encoder_layers).to(device)



        # --------------Decoder----------------

        decoder_layers_sizes = list(reversed(encoder_layers_sizes))

        decoder_layers = []                             # creates the decoder without the bottleneck
        for i in range(len(decoder_layers_sizes) - 1):
            decoder_layers.append(
                nn.Linear(decoder_layers_sizes[i], decoder_layers_sizes[i + 1])
                )
            decoder_layers.append(
                activation_fn()
                )
        self.amputated_decoder = nn.Sequential(*decoder_layers).to(device)


        # --------------Bottleneck----------------

        self.bottleneck_in = nn.Sequential(nn.Linear(encoder_layers_sizes[-1], latent_dim), bottleneck_in_fn())
        self.bottleneck_out = nn.Sequential(nn.Linear(latent_dim, decoder_layers_sizes[0]), activation_fn())


    # ----------------Encode, decode and Forward-------------------

    def encode(self, data):
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)  # data_flat has size (batch_size(64), 28, 28)
        else:
            data_flat = data
        return self.bottleneck_in(self.amputated_encoder(data_flat))


    def decode(self, data):
        return self.amputated_decoder(self.bottleneck_out(data))

    
    def forward(self, data): # batch has size (batch_size(64), 28, 28)
        original_shape = data.shape
        decoded = self.decode(self.encode(data))
        # Reshape decoded output back to original input shape
        if original_shape != decoded.shape:
            decoded = decoded.view(original_shape)
        return decoded

