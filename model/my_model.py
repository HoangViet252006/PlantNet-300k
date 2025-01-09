import math
import torch
from torch import nn
from model.flops_counter import get_model_complexity_info



model_params = {
    # width_expand, depth_expand, image_size, dropout_rate
    'model_0': [1.0, 1.0, 224, 0.2],
    'model_1': [1.0, 1.1, 240, 0.2],
    'model_2': [1.1, 1.2, 260, 0.3],
    'model_3': [1.2, 1.4, 280, 0.3],
    'model_4': [1.4, 1.8, 300, 0.3],
}

mb_block_settings = [
    # repeat| kernel_size| stride | expand | input | output |
    [1, 3, 1, 1, 16, 16],
    [2, 3, 2, 4, 16, 24],
    [2, 5, 2, 3, 24, 40],
    [3, 3, 2, 3, 40, 80],
    [3, 3, 1, 6, 80, 112],
    [4, 5, 2, 6, 112, 192],
    [1, 3, 1, 6, 192, 320],
]


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class ECA(nn.Module):
    """Constructs a ECA module.


    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out


    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()

        self.input_filters = in_channels
        self.output_filters = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.is_skip = True  # skip connection and drop connect

        # Expansion phase
        intermediate_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm2d(num_features=intermediate_channels)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=intermediate_channels, out_channels=intermediate_channels, groups=intermediate_channels,  # groups makes it depthwise
            kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=intermediate_channels)

        # Efficient Channel Attention
        self.eca = ECA(intermediate_channels)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self._bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.Hswish = nn.Hardswish(inplace=True)

    def forward(self, x, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self.Hswish(self._bn1(self._expand_conv(x)))
        x = self.Hswish(self._bn2(self._depthwise_conv(x)))

        # Efficient Channel Attention
        x = self.eca(x)

        x = self._bn3(self._project_conv(x))

        # Skip connection and drop connect
        if self.is_skip and self.stride == 1  and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x



class Model(nn.Module):
    def __init__(self, width_expand, depth_expand, num_classes, drop_connect_rate, dropout_rate):
        super(Model, self).__init__()

        self.drop_connect_rate = drop_connect_rate

        # Stem
        out_channels = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Hardswish(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            num_repeat, kernel_size, stride, expand_ratio, input_channels, output_channels = stage_setting
            # Update block input and output filters based on width multiplier.
            input_channels = input_channels if i == 0 else round_filters(input_channels, width_expand)
            output_channels = round_filters(output_channels, width_expand)
            num_repeat= num_repeat if i == 0 or i == len(mb_block_settings) - 1  else round_repeats(num_repeat, depth_expand)

            # The first block needs to take care of stride and filter size increase.
            stage.append(MBConvBlock(input_channels, output_channels, kernel_size, stride, expand_ratio))
            # if num_repeat > 1
            for _ in range(num_repeat - 1):
                stage.append(MBConvBlock(output_channels, output_channels, kernel_size, 1, expand_ratio))
            
            self.blocks.append(stage)

        # Head
        in_channels = round_filters(mb_block_settings[-1][5], width_expand)
        out_channels = 960
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Hardswish(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        out_channels_fc1 = 1280
        self.fc1 = nn.Linear(out_channels, out_channels_fc1)
        self.bn1 = nn.BatchNorm1d(out_channels_fc1)
        self.Hswish = nn.Hardswish(inplace=True)

        self.fc2 = nn.Linear(out_channels_fc1, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx += 1

        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout(self.Hswish(self.bn1(self.fc1(x))))
        else:
            x = self.Hswish(self.bn1(self.fc1(x)))

        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        

def build_model(name, num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = model_params[name]
    model = Model(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model

def calculate_latency(model_flops, device_gflops):
    latency = model_flops / device_gflops
    return latency


if __name__ == '__main__':
    model_name = 'model_4'
    model = build_model(model_name, 1081)

    model.eval()


    image_size = model_params[model_name][2]
    input_shape = (3, image_size, image_size)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')

    # Count all layers (including nested ones)
    layer_count = len(list(model.modules()))
    print(f"Number of all layers (including nested): {layer_count}")

    flops = flops.split(" ")[0]
    flops = float(flops)
    device_gflops = 2.13
    latency = calculate_latency(flops, device_gflops)
    print(f"Latency in IphoneX: {latency} second")

