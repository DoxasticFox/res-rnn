import torch

class Shift(torch.nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, s):
        x_original_width = x.size(-1)

        num_missing_dims = len(x.size()) - len(s.size())
        s = s.view((1,) * num_missing_dims + s.size())
        s = s.expand(
            x.size()[:num_missing_dims] +
            s.size()[num_missing_dims:]
        )

        x = torch.cat((s, x), dim=-1)
        x = x[..., :x_original_width]

        return x

class Res(torch.nn.Module):
    def __init__(self, width, identity_proportion=0.9):
        super(Res, self).__init__()

        self.identity_proportion = identity_proportion

        self.fc1  = torch.nn.Linear(width, width)
        self.fc2  = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight,  gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight,  gain=1.0)

    def forward(self, x):
        r = x
        x = self.fc1(x).abs()
        x = \
            r                              * self.identity_proportion + \
            (1 - self.identity_proportion) * self.fc2(x)
        return x

class ShiftedResNet(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_width,
            depth=50,
            identity_proportion=0.97
    ):
        super(ShiftedResNet, self).__init__()

        self.register_buffer('zero', torch.tensor([0.0]))

        self.fc1 = torch.nn.Linear(input_size, hidden_size)

        self.shf = torch.nn.ModuleList(
            Shift() for _ in range(depth))
        self.res = torch.nn.ModuleList(
            Res(hidden_size, identity_proportion) for _ in range(depth))

        self.fc2 = torch.nn.Linear(hidden_size, output_width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)

        for s, r in zip(self.shf, self.res):
            x = s(x, self.zero.expand((25,)))
            x = r(x)

        x = self.fc2(x)

        return x
