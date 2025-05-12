# file information
__author__ = 'Lily de Loe'
__date__ = '2024-10-16'

# import statements
from torch import nn
from torchvision import models
import torch

class WNet_Separate_VIIRS_Uncertainty(torch.nn.Module):

    def __init__(self, options):
        super().__init__()

        # 4 SAR channels (HH, HV, distance to coast, incidence angle) + 1 VIIRS channel (IST)
        self.input_block_v = DoubleConv(options, input_n=1,
                                      output_n=options['unet_conv_filters'][0])

        # encoding blocks for SAR, VIIRS
        self.contract_blocks_v = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks_v.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))

        # remaining channels correspond to AMSR2, ERA5, and aux
        self.input_block_arc = DoubleConv(options, input_n=len(options['train_variables'])-1,
                                      output_n=options['unet_conv_filters'][0])

        # encoding blocks for AMSR2, ERA5, and auxiliary
        self.contract_blocks_arc = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks_arc.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))

        # bridge between encoding branches and decoding branch
        # Note: the encoding branches are concatenated, doubling the number of input channels
        self.bridge = ContractingBlock(
            options, input_n=options['unet_conv_filters'][-1]*2, output_n=options['unet_conv_filters'][-1])

        # decoding blocks
        self.expand_blocks = torch.nn.ModuleList()
        # Note: this structure differs from unet.py, which wouldn't accurately sum the input channels from
        # the skip connections
        for expand_n in range(len(options['unet_conv_filters']), 0, -1): #range(star,t stop, step)

            if expand_n == len(options['unet_conv_filters']):
                self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n-1]*3,
                                                     output_n=options['unet_conv_filters'][expand_n - 1]))
            else:
                self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=(options['unet_conv_filters'][expand_n] + options['unet_conv_filters'][expand_n-1]*2),
                                                     output_n=options['unet_conv_filters'][expand_n - 1]))   
                                                                  
            #Note: revert to this code if something in the logic breaks when testing different filters
            #self.expand_blocks.append(ExpandingBlock(options=options,
            #                                         input_n=options['deconv_filters'][expand_n-1],
            #                                         output_n=options['unet_conv_filters'][expand_n - 1]))

        # regression layer + MSE loss
        self.regression_layer = torch.nn.Linear(options['unet_conv_filters'][0], 2)

        # 1x1 convolution + BCE loss
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])

    def forward(self, x):

        """Forward model pass."""

        # split the inputs (x) according to the two encoding paths
        # Note: review the data loader to view the order of input channel types
        x_viirs = x[:, -1:] #torch.cat((x[:, :4], x[:, -1:]), dim=1)
        x_ai4arctic = x[:,:-1]

        # encoding path for the SAR and VIIRS channels
        x_contract_v = [self.input_block_v(x_viirs)]
        for contract_block in self.contract_blocks_v:
            output_v = contract_block(x_contract_v[-1])
            x_contract_v.append(output_v)

        # encoding path for the AMSR2, ERA5, and auxiliary channels
        x_contract_arc = [self.input_block_arc(x_ai4arctic)] 
        for contract_block in self.contract_blocks_arc:
            output_arc = contract_block(x_contract_arc[-1])
            x_contract_arc.append(output_arc)

        # concetenate the tensors at each level of the SV and AEA branches.
        # Note: this is not really required because these lists are used separately for 
        # expand padding. This could be removed in future
        x_contract = [torch.cat((v, arc), dim=1) for v, arc in zip(x_contract_v, x_contract_arc)]

        # bridge connection between the encoding and decoding paths
        # Note: unlike the original U-Net, we have to reduce the number of channels by a factor
        # of 2 to account for concatenating the outputs of the SV and AEA branches
        x_expand = self.bridge(x_contract[-1])

        # decoding branch
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            # pass the current x, as well as the corresponding encoder layers
            x_expand = expand_block(x_expand, x_contract_v[up_idx - 1],x_contract_arc[up_idx - 1])
            up_idx -= 1

        # Predict mean and log-variance for SIC
        sic_outputs = self.regression_layer(x_expand.permute(0, 2, 3, 1))
        sic_mean = sic_outputs[..., 0]  # First output is mean
        sic_log_variance = sic_outputs[..., 1]  # Second output is log-variance
        sic_variance = torch.exp(sic_log_variance)  # Convert log-variance to variance

        return {'SIC': {'mean': sic_mean, 'variance': sic_variance},
                'SOD': self.sod_feature_map(x_expand),
                'FLOE': self.floe_feature_map(x_expand)}


class FeatureMap(torch.nn.Module):

    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)

class DoubleConv(torch.nn.Module):

    """Class to create a double convolutional layer."""

    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_n,
                            out_channels=output_n,
                            kernel_size=options['conv_kernel_size'],
                            stride=options['conv_stride_rate'],
                            padding=options['conv_padding'],
                            padding_mode=options['conv_padding_style'],
                            bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_n,
                            out_channels=output_n,
                            kernel_size=options['conv_kernel_size'],
                            stride=options['conv_stride_rate'],
                            padding=options['conv_padding'],
                            padding_mode=options['conv_padding_style'],
                            bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):

        """Pass x through the double convolutional layer."""

        x = self.double_conv(x)

        return x

class ContractingBlock(torch.nn.Module):
    
    """Class to perform downward pass in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x

class ExpandingBlock(torch.nn.Module):

    """
    Class to perform upward layer in the U-Net.
    Note: this class differs from its namesake in unet.py
    """

    def __init__(self, options, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['conv_padding_style']
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n, output_n=output_n)

    def forward(self, x, x_v, x_arc):

        """Pass x through the upward layer and concatenate with the opposite layers from the 
        two encoding paths."""

        x = self.upsample(x)

        # ensure that x_decoder and x_encoder match for all three branches
        x = expand_padding(x, x_v, x_arc, padding_style=self.padding_style)

        x = torch.cat([x, x_v, x_arc], dim=1)

        return self.double_conv(x)

def expand_padding(x, x_v, x_arc, padding_style: str = 'constant'):

    """
    Ensure that x_decoder and x_encoder match for all three branches
    Note: this function differs from its namesake in unet.py

    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_sv :
        Image tensor of shape (batch size, channels, height, width). Contracting path for 
        SAR and VIIRS inputs.
    x_aea :
        Image tensor of shape (batch size, channels, height, width). Contracting path for 
        AMSR2, ERA5, and auxiliary inputs.    
    padding_style : str
        Type of padding.

    Returns
    -------
    x : Padded tensor for the expanding/decoding path.
    """

    # Check whether x_sv/x_aea is tensor or shape.
    if (type(x_v) == type(x)) and (type(x_arc) == type(x)):
        # Check that the two tensors are the same size
        if x_v.size() == x_arc.size():
            x_contract = x_v.size()
        else:
            print("ERROR: mismatched size")
    else:
        print("ERROR: mismatched type")

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x