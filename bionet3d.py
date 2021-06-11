from torch.nn import Module, Sequential, Conv3d, BatchNorm3d, ConvTranspose3d, ReLU, MaxPool3d, Sigmoid, Parameter
from torch import tensor, cat


class BiONet(Module):

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 1,
                 iterations: int = 2,
                 multiplier: float = 1.0,
                 num_layers: int = 4,
                 integrate: bool = False):

        super(BiONet, self).__init__()
        #  
        self.input_channels = input_channels
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01
        self.filters_list = [int(32 * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]

        # First downsizing block
        self.pre_transform_conv_block = Sequential(
            Conv3d(self.input_channels, self.filters_list[0], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1), 
            ReLU(),
            BatchNorm3d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv3d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv3d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[0], momentum=self.batch_norm_momentum),
            MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=(0, 0, 0))
        )
        
        self.reuse_convs = []  
        self.encoders = []  
        self.reuse_deconvs = []
        self.decoders = []
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):


                in_channel = self.filters_list[layer] * 2
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                #  Encoder
                if iteration == 0:
                    conv1 = Conv3d(in_channel, mid_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1)
                    conv2 = Conv3d(mid_channel, mid_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1)
                    conv3 = Conv3d(mid_channel, out_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1)
                    self.reuse_convs.append((conv1, conv2, conv3))

                convs = Sequential(
                    self.reuse_convs[layer][0],
                    ReLU(),
                    BatchNorm3d(mid_channel, momentum=self.batch_norm_momentum),
                    self.reuse_convs[layer][1],
                    ReLU(),
                    BatchNorm3d(mid_channel, momentum=self.batch_norm_momentum)
                )
                #  DOWN
                down = Sequential(
                    self.reuse_convs[layer][2],
                    ReLU(),
                    BatchNorm3d(out_channel, momentum=self.batch_norm_momentum),
                    MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=(0, 0, 0))
                )
                self.add_module("iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((convs, down))

                #  Decoders
                in_channel = self.filters_list[self.num_layers - layer] + self.filters_list[self.num_layers - 1 - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
               
                if iteration == 0:
                    conv1 = Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1)
                    conv2 = Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1)
                    conv3 = ConvTranspose3d(out_channel, out_channel, kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=2, output_padding=(1, 1, 1))
                    self.reuse_deconvs.append((conv1, conv2, conv3))

                convs = Sequential(
                    self.reuse_deconvs[layer][0],
                    ReLU(),
                    BatchNorm3d(out_channel, momentum=self.batch_norm_momentum),
                    self.reuse_deconvs[layer][1],
                    ReLU(),
                    BatchNorm3d(out_channel, momentum=self.batch_norm_momentum)
                )
                # UP
                up = Sequential(
                    self.reuse_deconvs[layer][2],
                    ReLU(),
                    BatchNorm3d(out_channel, momentum=self.batch_norm_momentum)
                )
                self.add_module("iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.decoders.append((convs, up))
        #  Bottleneck
        self.middles = Sequential(
            Conv3d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            Conv3d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ConvTranspose3d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3, 3),padding=(1, 1, 1),
                            stride=2, output_padding=(1, 1, 1)),
            ReLU(),
            BatchNorm3d(self.filters_list[-1], momentum=self.batch_norm_momentum)
        )
        self.post_transform_conv_block = Sequential(
            Conv3d(self.filters_list[0] * self.iterations, self.filters_list[0], kernel_size=(3, 3, 3),padding=(1, 1, 1),
                   stride=1) if self.integrate else Conv3d(self.filters_list[0],
                                                                self.filters_list[0], kernel_size=(3, 3, 3),
                                                               padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv3d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3, 3),padding=(1, 1, 1), stride=1),
            ReLU(),
            BatchNorm3d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv3d(self.filters_list[0], 1, kernel_size=(1, 1, 1), stride=1),
            Sigmoid(),
        )

    def forward(self, x: tensor) -> tensor:
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        all_output = [None for i in range(self.iterations)]
        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            
            # Through encoder
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x
#                 print(x_in.shape)
                x_in = self.encoders[e_i][0](cat([x_in, x_in if dec[-1 - layer] is None else dec[-1 - layer]], dim=1))
                enc[layer] = x_in
                x_in = self.encoders[e_i][1](x_in)
                e_i = e_i + 1
                
            # Bottleneck
            x_in = self.middles(x_in)
#             print(x_in.shape)
            
            # Decoder
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i][0](cat([x_in, enc[-1 - layer]], dim=1))
                dec[layer] = x_in
                x_in = self.decoders[d_i][1](x_in)
                d_i = d_i + 1
                
            all_output[iteration] = x_in
        if self.integrate:
            x_in = cat(all_output, dim=1)
        x_in = self.post_transform_conv_block(x_in)
        return x_in