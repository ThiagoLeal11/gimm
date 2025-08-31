import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEncoder(nn.Module):
    def __init__(self, img_size, n_channels, patch_size=4, projection_ouput_size=None, overlap=0, dropout_rate=0.0, stride_size=None, **kwargs):
        """
        Encodes an image to a vector according to ViT process: patches, projection, cls token and positional embedding
        Uses standard PyTorch modules (nn.Conv2d, nn.Flatten, nn.LayerNorm) instead of custom implementations
        :param img_size: input images size, the image must be square sized
        :param n_channels: number of channel in the input images
        :param patch_size: size of each patches, patches will be square sized
        :param projection_ouput_size: number of feature at the output of the projection
        :param overlap: number of overlapping pixels for neighbouring patches (deprecated, use stride_size)
        :param stride_size: stride for patch extraction
        :param dropout_rate: dropout rate at the final stage level
        """
        super(PatchEncoder, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.stride_size = stride_size if stride_size is not None else patch_size

        self.proj_output_size = projection_ouput_size if projection_ouput_size is not None else (n_channels * patch_size * patch_size)

        # Replace custom projection with standard PyTorch modules
        self.patch_conv = nn.Conv2d(
            n_channels, 
            self.proj_output_size, 
            kernel_size=patch_size, 
            stride=self.stride_size, 
            padding=0
        )
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        self.flatten = nn.Flatten(start_dim=2)  # Flatten H and W dimensions
        
        self.layer_norm = nn.LayerNorm(self.proj_output_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_output_size))

        # Calculate the number of patches correctly
        num_patches = ((img_size - patch_size + self.stride_size) // self.stride_size) ** 2 + 1
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, self.proj_output_size))

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, imgs):
        assert len(imgs.shape) == 4, 'Expected input image tensor to be of shape BxCxHxW'
        b, _, _, _ = imgs.shape

        # Project to patches using standard Conv2d
        x = self.patch_conv(imgs)  # Shape: (B, proj_output_size, H_patches, W_patches)
        
        # Flatten and transpose to get (B, num_patches, proj_output_size)
        x = self.flatten(x)  # Shape: (B, proj_output_size, H_patches * W_patches)
        x = x.transpose(1, 2)  # Shape: (B, H_patches * W_patches, proj_output_size)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        # Add cls token
        cls_tokens = self.cls_token.expand(b, 1, self.proj_output_size)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding with sine activation
        x += torch.sin(self.pos_embedding)

        return self.dropout(x)


if __name__ == '__main__':
    fim = torch.randn(100, 3, 32, 32)
    e = PatchEncoder(32, 3, 8, 384, 0)
    result = e.forward(fim)
    print(result.shape)
