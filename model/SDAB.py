import torch
import torch.nn as nn

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module, a=0, mode='fan_out', nonnonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonnonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonnonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

class ResBlock(nn.Module):

    def __init__(self, inplanes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out

class RDBlock(nn.Module):
    def __init__(self, inplanes):
        super(RDBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(2*inplanes, inplanes, 3, 1, 1)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = torch.cat([x, out], 1)
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out

class Attention(nn.Module):
    def __init__(self, inplanes):
        super(Attention, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.relu2 = nn.PReLU()

        self.conv_mask = nn.Conv2d(inplanes, 1, 3, 1, 1)

        kaiming_init(self.conv_mask, mode='fan_in')

        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=1),
            nn.LayerNorm([inplanes // 2, 1, 1]),
            nn.PReLU(),
            nn.Conv2d(inplanes // 2, inplanes, kernel_size=1)
        )
        last_zero_init(self.channel_add_conv)

        self.d_conv1 = nn.Conv2d(inplanes, 1, 3, 1, 1)
        self.d_conv2 = nn.Conv2d(inplanes, 1, 3, 1, 2, dilation=2)
        self.d_conv4 = nn.Conv2d(inplanes, 1, 3, 1, 3, dilation=3)


        self.fusion = nn.Conv2d(3, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.fine = nn.Conv2d(2*inplanes, inplanes, 1)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)

        input_x = input_x.unsqueeze(1)

        context_mask = self.conv_mask(x)

        context_mask = context_mask.view(batch, 1, height * width)

        context_mask = self.softmax(context_mask)

        context_mask = context_mask.unsqueeze(3)

        context = torch.matmul(input_x, context_mask)

        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)

        convtext = self.spatial_pool(out)
        convtext = self.channel_add_conv(convtext)

        out = out + convtext

        ff1 = self.d_conv1(out)
        ff2 = self.d_conv2(out)
        ff4 = self.d_conv4(out)
        ff = torch.cat([ff1, ff2, ff4], 1)
        sp_mask = self.sigmoid(self.fusion(ff))
        sp_feature = out * sp_mask

        sp_feature = torch.cat([out, sp_feature], 1)
        sp_feature = self.fine(sp_feature)
        sp_feature += x
        sp_feature = self.relu2(sp_feature)
        return sp_feature

class share_weight(nn.Module):
    def __init__(self, inplanes):
        super(share_weight, self).__init__()
        self.bef_res = nn.Sequential(
            RDBlock(inplanes),
            RDBlock(inplanes),
            RDBlock(inplanes)
        )
        self.att = nn.Sequential(
            Attention(inplanes),
            Attention(inplanes),
            Attention(inplanes),
            Attention(inplanes),
        )
        self.lat_res = nn.Sequential(
            RDBlock(inplanes),
            RDBlock(inplanes),
        )
        self.dilconv1 = nn.Conv2d(inplanes, int(inplanes/4), 3, 1, 1)
        self.dilconv2 = nn.Conv2d(inplanes, int(inplanes/4), 3, 1, 2, dilation=2)
        self.dilconv4 = nn.Conv2d(inplanes, int(inplanes/4), 3, 1, 3, dilation=3)
        self.dilconv8 = nn.Conv2d(inplanes, int(inplanes/4), 3, 1, 4, dilation=4)
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
    def forward(self, x):
        out = self.bef_res(x)
        out = self.att(out)
        out = self.lat_res(out)

        h1 = self.dilconv1(out)
        h2 = self.dilconv2(out)
        h4 = self.dilconv4(out)
        h8 = self.dilconv8(out)
        mix_feature = torch.cat([h1, h2, h4, h8, out], 1)
        mix_feature = self.fusion(mix_feature)
        return mix_feature

class SDAB_4(nn.Module):
    def __init__(self):
        super(SDAB_4, self).__init__()
        self.fea_in_f = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.PReLU())
        self.fea_in_s = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.PReLU())

        self.main_weight = share_weight(32)

        self.last_conv = nn.Conv2d(3, 3, 3, 1, 1)


    def forward(self, x):
        original = x
        outs = list()

        out = self.fea_in_f(x)
        skip_concat = out

        out = self.main_weight(out)
        outs.append(out)

        out = torch.cat([out, x], 1)
        out = self.fea_in_s(out) + skip_concat
        out = self.main_weight(out)
        outs.append(out)

        out = original - out

        out = self.last_conv(out) + out
        outs.append(out)

        return outs

