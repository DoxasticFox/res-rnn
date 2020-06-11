import torch

def _broadcast_but_last(x, y):
    if len(x.size()) == len(y.size()):
        return x, y

    if len(x.size()) > len(y.size()):
        bigger, smaller = x, y
    else:
        bigger, smaller = y, x

    new = smaller.view(
        (1,) * (len(bigger.size()) - len(smaller.size())) + \
        smaller.size()
    )
    new = new.expand(
        bigger.size()[:-len(smaller.size())] + \
        smaller.size()
    )

    if len(x.size()) > len(y.size()):
        return x, new
    else:
        return new, y

class Affine(torch.nn.Module):
    def __init__(self, width):
        super(Affine, self).__init__()

        self.w = torch.nn.Parameter(torch.ones(width))
        self.b = torch.nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return self.w * x + self.b

class Shift(torch.nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, s):
        x_original_width = x.size(-1)

        x, s = _broadcast_but_last(x, s)
        x = torch.cat((s, x), dim=-1)
        x = x[..., :x_original_width]

        return x

class Overwrite(torch.nn.Module):
    def __init__(self):
        super(Overwrite, self).__init__()

    def forward(self, x, o):
        o_original_width = o.size(-1)

        x, o = _broadcast_but_last(x, o)
        x = torch.cat((o, x[..., o_original_width:]), dim=-1)

        return x


class BatchNorm(torch.nn.Module):
    def __init__(self, width):
        super(BatchNorm, self).__init__()

        self.smooth_l1_loss = torch.nn.SmoothL1Loss()

        self.register_buffer('zeros', torch.zeros(width))
        self.register_buffer('ones', torch.ones(width))

    def forward(self, x):
        assert(len(x.size()) == 2)

        mean_reg = self.smooth_l1_loss(x.mean(0), self.zeros)
        var_reg  = self.smooth_l1_loss(x.var(0), self.ones)
        reg = mean_reg + var_reg

        return x, reg

class Res(torch.nn.Module):
    def __init__(self, width, identity_proportion):
        super(Res, self).__init__()

        self.identity_proportion = identity_proportion

        self.bn   = BatchNorm(width)
        self.fc1  = torch.nn.Linear(width, width)
        self.fc2  = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x

        x, bn_reg = self.bn(x)

        x = self.fc1(x)
        x = x.abs()
        x = self.fc2(x)

        x = self.identity_proportion * r + (1 - self.identity_proportion) * x

        return x, bn_reg

class ShiftedResNet(torch.nn.Module):
    def __init__(
            self,
            input_width,
            hidden_width,
            output_width,
            depth=100,
            identity_proportion=0.97
    ):
        super(ShiftedResNet, self).__init__()

        self.register_buffer('zero', torch.tensor([0.0]))

        self.fc1 = torch.nn.Linear(input_width, hidden_width)

        self.shf = torch.nn.ModuleList(
            Shift() for _ in range(depth))
        self.res = torch.nn.ModuleList(
            Res(hidden_width, identity_proportion) for _ in range(depth))

        self.fc2 = torch.nn.Linear(hidden_width, output_width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)

        for s, r in zip(self.shf, self.res):
            x = s(x, self.zero.expand((25,)))
            x = r(x)

        x = self.fc2(x)

        return x

class ResRnn(torch.nn.Module):
    def __init__(self, input_width, state_width, output_width, identity_proportion=0.97):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.stream_width = input_width + state_width
        self.identity_proportion = identity_proportion

        assert(output_width <= self.stream_width)

        self.ow = Overwrite()
        self.aff = Affine(self.stream_width)
        self.res = Res(self.stream_width, self.identity_proportion)

    def forward(self, input):
        # input:         (seq_width, batch_width, input_width)
        # output_stream: (seq_width, batch_width, input_width + state_width)
        # returns:       (batch_width, output_width)

        assert(len(input.size()) == 3)
        assert(input.size(-1) == self.input_width)

        with torch.no_grad():
            output_stream = torch.normal(
                0.0,
                1.0,
                size=(input.size(1), self.stream_width)
            ).to(input.device)

        reg_terms = []
        for index, element in enumerate(input):
            output_stream = self.ow(output_stream, element)
            if index == 0:
                output_stream = self.aff(output_stream)
            output_stream, reg_term = self.res(output_stream)

            reg_terms.append(reg_term)

        # Truncate the output of the last RNN application. We take the last
        # output elements instead of the first because we hypothesise that this
        # will make learning long distance dependencies easier.
        output_stream = output_stream[..., -self.output_width:]

        reg = torch.stack(reg_terms).mean()

        return output_stream, reg
