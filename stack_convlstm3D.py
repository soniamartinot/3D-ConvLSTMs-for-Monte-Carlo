from convlstm3D import *

def stack_model_3D(keep_in_memory=False):
    convlstm = ConvLSTM3D(input_dim=1,
                 hidden_dim=[32, 64, 128, 32, 1],
                 kernel_size=(3, 3, 3),
                 num_layers=5,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 keep_in_memory=keep_in_memory)
    return convlstm

def stack_model_3D_deep():
    convlstm = ConvLSTM3D(input_dim=1,
                 hidden_dim=[32, 64, 128, 64, 64, 32, 1],
                 kernel_size=(3, 3, 3),
                 num_layers=7,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
    return convlstm

