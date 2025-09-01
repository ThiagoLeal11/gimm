import torch
import torch.nn as nn

from gimm.models.utils import count_params
from gimm.models.xvitgan.components.patch_encoder import PatchEncoder
from gimm.models.xvitgan.components.transformer import Transformer
from gimm.models.xvitgan.components.mlp import MLP


class Discriminator(nn.Module):
    def __init__(self, img_size, n_channels, output_size, n_transformer_layers=1, encoder_params=None, transformer_params=None, mlp_params=None, **kwargs):
        """
        Discriminator module for ViTGAN model
        :param img_size: input images size, the image must be square sized
        :param n_channels: number of channel in the input images
        :param output_size: number of output features per input image
        :param n_transformer_layers: number of stacked transformer blocks
        :param encoder_params: kwargs for optional parameters of the PatchEncoder, mandatory args will be filled automatically
        :param transformer_params: kwargs for optional parameters for each Transformer block, mandatory args will be filled automatically
        :param mlp_params: kwargs for optional parameters of the output MLP module, mandatory args will be filled automatically
        """
        super(Discriminator, self).__init__()

        self.img_size             = img_size
        self.n_channels           = n_channels
        self.output_size          = output_size
        self.n_transformer_layers = n_transformer_layers

        self.encoder_params     = {} if encoder_params is None else encoder_params
        self.transformer_params = {} if transformer_params is None else transformer_params
        self.mlp_params         = {} if mlp_params is None else mlp_params

        # Set correct parameters for discriminator
        self.encoder_params['img_size'], self.encoder_params['n_channels'] = self.img_size, self.n_channels

        # Use correct patch configuration like in notebook
        if 'patch_size' not in self.encoder_params:
            self.encoder_params['patch_size'] = 8  # 2x the generator patch size for overlap
        if 'stride_size' not in self.encoder_params:
            self.encoder_params['stride_size'] = 4  # Same as generator patch size

        self.patch_encoder = PatchEncoder(**self.encoder_params)

        # Set discriminator parameters for transformer
        self.transformer_params['in_features'] = self.patch_encoder.proj_output_size
        self.transformer_params['spectral_scaling'] = True
        self.transformer_params['lp'] = 2  # L2 distance for discriminator

        self.transformer_layers = nn.ModuleList([Transformer(**self.transformer_params) for _ in range(self.n_transformer_layers)])

        self.mlp_params['in_features'], self.mlp_params['out_features'] = self.transformer_layers[-1].in_features, self.output_size
        self.mlp = MLP(**self.mlp_params)

        self.sigmoid = torch.nn.Sigmoid()

        print(f'Discriminator model with {count_params(self)} parameters ready')

    def forward(self, imgs):
        tokens = self.patch_encoder(imgs)
        for transformer in self.transformer_layers:
            tokens = transformer(tokens)
        output = self.mlp(tokens[:, 0, :])  # we compute the output only with the cls token
        return self.sigmoid(output)


if __name__ == '__main__':
    x = torch.randn(100, 3, 64, 64)
    d = Discriminator(64, 3, 2, 2)
    o = d(x)
    print(o.shape)
