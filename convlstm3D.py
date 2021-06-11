import torch.nn as nn
import torch


class ConvLSTM3DCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, keep_in_memory=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM3DCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias
        
        # This is just for interpretation of the inner works of the cell. Nothing to do with training.
        self.keep_in_memory = keep_in_memory
        if self.keep_in_memory:
            self.memory = {"h_cur":[],
                           "c_cur":[],
                           "i":[],
                           "f":[],
                           "o":[],
                           "g":[],
                           "c_next":[],
                           "h_next":[]}
        
        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i) # Input gate 
        f = torch.sigmoid(cc_f) # forget gate 
        o = torch.sigmoid(cc_o) # output gate 
        g = torch.tanh(cc_g)    # New candidate values that could be added to the state

        c_next = f * c_cur + i * g # New cell state
        h_next = o * torch.tanh(c_next) # Output of the cell        
        
        if self.keep_in_memory:  
            self.memory['h_cur'] += [h_cur.cpu().detach().numpy()]
            self.memory['c_cur'] += [c_cur.cpu().detach().numpy()]
            self.memory['i'] += [i.cpu().detach().numpy()]
            self.memory['f'] += [f.cpu().detach().numpy()]
            self.memory['o'] += [o.cpu().detach().numpy()]
            self.memory['g'] += [g.cpu().detach().numpy()]
            self.memory['c_next'] += [c_next.cpu().detach().numpy()]
            self.memory['h_next'] += [h_next.cpu().detach().numpy()]        
        return h_next, c_next
    

    def init_hidden(self, batch_size, image_size):
        height, width, depth = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, depth, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, depth, device=self.conv.weight.device))


class ConvLSTM3D(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
    Input:
        A tensor of size B, T, C, H, W, D or T, B, C, H, W, D
    Output:
        The last output of the last layer.
        OR
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory

    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, keep_in_memory=False):
        super(ConvLSTM3D, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.keep_in_memory = keep_in_memory

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTM3DCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                           keep_in_memory=self.keep_in_memory))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            6-D Tensor either of shape (t, b, c, h, w, d) or (b, t, c, h, w, d)
        Returns
        -------
        last_state_list, layer_output or last_state_list[-1][-1] (last output of last layer)
        """
        if not self.batch_first:
            # (t, b, c, h, w, d) -> (b, t, c, h, w, d)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

        b, _, _, h, w, d = input_tensor.size()

        # Initialize hidden state
        hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w, d))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            # Iterate over timesteps
            for t in range(seq_len):
                # Do forward
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h) # List of hidden states of each layer for this timestep

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if self.return_all_layers:            
            return layer_output_list, last_state_list
        else:
            # Return final output of last lauer
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            return last_state_list[-1][-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param