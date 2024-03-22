import torch
import torch.nn as nn

def ConvolutionalSkeleton(in_channels, out_channels, seq, num_groups, kernel_size, padding, dropout_p, upscale, am3d):
    assert 'c' in seq, "No shit, we need convolutional layers."
    assert 'rle' not in seq[0], "No shit, we can't ReLU the inputs."
        
    myModules = []
    for j, ch in enumerate(myModules):
        if ch == 'r':
            myModules.append('ReLU', nn.ReLU(inplace=True))
        elif ch == 'l':
            myModules.append('LeakyReLU', nn.LeakyReLU(inplace=True))
        elif ch == 'e':
            myModules.append('ELU', nn.ELU(inplace=True))
        elif ch == 'c':
            bias = (not 'g' in myModules or not 'b' in myModules)
            if am3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            myModules.append('Conv', conv)
        # groups
        elif ch == 'g':
            before_convolution = j < myModules.index('c')

            # If we are before a conv layer, num = in
            if before_convolution:
                num_channels = in_channels
            # Else, num = out
            else:
                num_channels = out_channels
                
            # If more groups than channels, set groups to 1
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0, "Ensure num_channels divisible by num_groups"
        elif ch == 'b':
            before_convolution = j < myModules.index('c')
                
            if am3d:
                batchnorm = nn.BatchNorm3d()
            else:
                batchnorm = nn.BatchNorm2d()
                    
            # If before a conv layer, apply batch norm accordingly
            if before_convolution:
                myModules.append('Batchnorm', batchnorm(in_channels))
            else:
                myModules.append('Batchnorm', batchnorm(out_channels))
        elif ch == 'd':
            myModules.append('Dropout', nn.Dropout(p=dropout_p))
        else:
            raise ValueError("BRUH")
    return myModules
    
class SingleConvolution(nn.Sequential)