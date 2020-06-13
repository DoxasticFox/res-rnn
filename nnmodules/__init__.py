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

class Shift(torch.nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, s):
        x_original_width = x.size(-1)

        x, s = _broadcast_but_last(x, s)
        x = torch.cat((s, x), dim=-1)
        x = x[..., :x_original_width]

        return x

class Res(torch.nn.Module):
    def __init__(self, width, linearity):
        super(Res, self).__init__()

        self.linearity = linearity

        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x

        x = self.fc1(x)
        x = x.clamp(min=0)
        x = self.fc2(x)

        x = self.linearity * r + (1 - self.linearity) * x

        return x

class ShiftedResNet(torch.nn.Module):
    def __init__(
            self,
            input_width,
            hidden_width,
            output_width,
            depth=100,
            linearity=0.97
    ):
        super(ShiftedResNet, self).__init__()

        self.register_buffer('zero', torch.tensor([0.0]))

        self.fc1 = torch.nn.Linear(input_width, hidden_width)

        self.shf = torch.nn.ModuleList(
            Shift() for _ in range(depth))
        self.res = torch.nn.ModuleList(
            Res(hidden_width, linearity) for _ in range(depth))

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
    def __init__(self, input_width, state_width, output_width, linearity):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.stream_width = input_width + state_width
        self.linearity = linearity

        assert(output_width <= self.stream_width)

        self.register_buffer('zero', torch.tensor([0.0]))

        self.ins = Shift()
        self.res = Res(self.stream_width, self.linearity)

    def forward(self, input):
        # input:         (seq_width, batch_width, input_width)
        # output_stream: (seq_width, batch_width, input_width + state_width)
        # returns:       (batch_width, output_width)

        assert(len(input.size()) == 3)
        assert(input.size(-1) == self.input_width)

        output_stream = self.zero.expand(self.stream_width)

        for index, element in enumerate(input):
            output_stream = self.ins(output_stream, element)
            output_stream = self.res(output_stream)

        # Truncate the output of the last RNN application. We take the last
        # output elements instead of the first because we hypothesise that this
        # will make learning long distance dependencies easier.
        output_stream = output_stream[..., -self.output_width:]


        return output_stream
