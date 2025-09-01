import torch
import torch.nn as nn

from gimm.models.utils import count_params
from gimm.models.xvitgan.components.mlp import MLP
from gimm.models.xvitgan.components.transformer import TransformerSLN
from gimm.models.xvitgan.components.sln import SLN
from gimm.models.xvitgan.components.siren import SIREN


class Generator(nn.Module):
    def __init__(self, lattent_size, img_size, n_channels, feature_hidden_size=384, n_transformer_layers=1, output_hidden_dim=768, mapping_mlp_params=None, transformer_params=None, **kwargs):
        """
        ViT Generator Class
        :param lattent_size: number of features in the lattent space
        :param img_size: output images size, the image will be square sized
        :param n_channels: number of channel in the output images
        :param feature_hidden_size: number of features in the transformers and output layers
        :param n_transformer_layers: number of stacked transformer blocks
        :param mapping_mlp_params: kwargs for optional parameters of the mapping MLP, mandatory args will be filled automatically
        :param transformer_params: kwargs for optional parameters of the Transformer blocks, mandatory args will be filled automatically
        """
        super(Generator, self).__init__()

        self.lattent_size          = lattent_size
        self.img_size              = img_size
        self.feature_hidden_size   = feature_hidden_size
        self.n_channels            = n_channels
        self.n_transformer_layers  = n_transformer_layers
        self.output_hidden_dim     = output_hidden_dim

        self.mapping_params        = {} if mapping_mlp_params is None else mapping_mlp_params
        self.transformer_params    = {} if transformer_params is None else transformer_params

        # Patch size for generator (original implementation uses 4)
        self.patch_size = 4
        self.num_patches = (img_size // self.patch_size) ** 2

        # map latent vector to concatenated patch embeddings
        self.mapping_params['in_features'], self.mapping_params['out_features'] = self.lattent_size, self.num_patches * self.feature_hidden_size
        self.mapping_mlp = MLP(**self.mapping_params)

        self.pos_emb = nn.Parameter(torch.randn(self.num_patches, self.feature_hidden_size))

        self.transformer_params['in_features'], self.transformer_params['spectral_scaling'], self.transformer_params['lp'] = self.feature_hidden_size, False, 1
        self.transformer_layers = nn.ModuleList([TransformerSLN(**self.transformer_params) for _ in range(self.n_transformer_layers)])

        self.sln = SLN(self.feature_hidden_size)

        self.output_net = nn.Sequential(
            SIREN(self.feature_hidden_size, output_hidden_dim, is_first=True),
            # output per patch: n_channels * patch_size * patch_size
            SIREN(output_hidden_dim, self.n_channels * self.patch_size * self.patch_size, is_first=False)
        )

        print(f'Generator model with {count_params(self)} parameters ready')

    def forward(self, x):
        # reshape mapping output to [batch, num_patches, feature_hidden_size]
        batch_size = x.shape[0]
        w = self.mapping_mlp(x).view(batch_size, self.num_patches, self.feature_hidden_size)
        # initialize positional embeddings and expand to batch size
        h = self.pos_emb.unsqueeze(0).expand(batch_size, self.num_patches, self.feature_hidden_size)
        for tf in self.transformer_layers:
            w, h = tf(h, w)
        w = self.sln(h, w)
        res = self.output_net(w).view(x.shape[0], self.n_channels, self.img_size, self.img_size)
        return res


if __name__ == '__main__':
    ipt = torch.randn(10, 1024)
    mod = Generator(1024, 64, 3, 200)
    ret = mod.forward(ipt)
    print(ret.shape)
