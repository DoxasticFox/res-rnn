import __main__
import collections
import itertools
import os
import torch

class Res(torch.nn.Module):
    def __init__(self, width, linearity):
        super(Res, self).__init__()

        assert(width > 0)
        assert(0.0 <= linearity < 1.0)

        self.linearity = linearity

        self.randperm = torch.nn.Parameter(
            torch.randperm(width),
            requires_grad=False,
        )
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x[..., self.randperm]

        r = x

        x = self.fc1(x)
        x = x.clamp(min=0)
        x = self.fc2(x)

        x = self.linearity * r + (1 - self.linearity) * x

        return x

class ResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        stream_width,
        output_width,
        linearity=0.99999,
        checkpoint_name='ResRnn',
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.stream_width = stream_width
        self.output_width = output_width
        self.linearity = linearity
        self.checkpoint_name = checkpoint_name

        assert(self.input_width <= self.stream_width)
        assert(self.output_width <= self.stream_width)
        assert(self.input_width >= 0)
        assert(self.stream_width >= 0)
        assert(self.output_width >= 0)

        # Variables for coordinating checkpointing
        self.checkpoint_dir = None
        self.num_checkpoints = 0

        self.initial_stream = torch.nn.Parameter(
            torch.zeros(self.stream_width),
        )

        self.res = Res(self.stream_width, self.linearity)

    def forward(self, input, stream=None, seq_indices=-1):
        # input:
        #     (seq_width, batch_width, input_width)
        # stream:
        #     (batch_width, stream_width) or
        #     (stream_width)
        # seq_indices:
        #     int or
        #     [int]
        # returns:
        #     (seq_width, batch_width, output_width) and
        #     (seq_width, batch_width, stream_width) or
        #     (           batch_width, output_width) and
        #     (           batch_width, stream_width) or

        # Validate inputs
        input_seq_width, input_batch_width, input_width = input.size()

        stream_batch_width, stream_width = None, None
        if stream is not None:
            try:
                (stream_width,) = stream.size()
            except ValueError:
                pass

            try:
                stream_batch_width, stream_width = stream.size()
            except ValueError:
                pass

        assert(input_width == self.input_width)
        assert(stream is None or stream_width == self.stream_width)
        assert(
            stream_batch_width is None or
            stream_batch_width == input_batch_width)
        assert(input_batch_width > 0)
        assert(
            type(seq_indices) is not torch.Tensor or
            seq_indices.size() == (input_batch_width,)
        )

        # Set initial stream
        if stream is None:
            stream = (1 - self.linearity) * self.initial_stream

        # Pad input
        input_padding = torch.zeros(
            (input.size(0), input.size(1), self.stream_width - input.size(2)),
            device=input.device,
        )
        padded_input = torch.cat((input, input_padding), dim=2)

        # Apply RNN
        streams = []
        for element in padded_input:
            stream = self.res(stream + element)
            streams.append(stream)

        # Stack streams and ensure everything is going as planned
        streams = torch.stack(streams)
        assert(
            streams.size() ==
                (input_seq_width, input_batch_width, self.stream_width)
        )

        # We take the last elements as the output instead of the first because
        # we hypothesise that this will make learning long distance dependencies
        # easier. We slice it in a verbose way to allow for zero-length slices.
        seq_indices = (
            slice(None, None, None)
            if seq_indices is None
            else seq_indices
        )
        outputs = streams[
            seq_indices,
            torch.arange(input_batch_width),
            self.stream_width - self.output_width:self.stream_width
        ]

        streams = streams[
            seq_indices,
            torch.arange(input_batch_width),
            :
        ]

        return outputs, streams

    def _set_checkpoint_dir_if_none(self, file_name):
        def candidate_paths(path_name):
            yield path_name
            yield from (path_name + '-' + str(i + 2) for i in itertools.count())
        def unique_path(path_name):
            for candidate_path in candidate_paths(path_name):
                if not os.path.exists(candidate_path):
                    return candidate_path

        # Set self.checkpoint_dir if it isn't already
        if self.checkpoint_dir is None:
            abs_save_dir = os.path.dirname(os.path.realpath(__main__.__file__))
            abs_file_name = (
                file_name
                if os.path.isabs(file_name)
                else os.path.join(abs_save_dir, file_name)
            )
            self.checkpoint_dir = unique_path(os.path.dirname(abs_file_name))

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

    def save_checkpoint(self):
        file_name = 'checkpoints/{}/{}.pt'.format(
            self.checkpoint_name,
            self.num_checkpoints
        )

        self._set_checkpoint_dir_if_none(file_name)

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

        print(
            'Saving checkpoint for {} to {}'.format(
                self.checkpoint_name,
                checkpoint_file_name
            )
        )

        # Create dir + checkpoint
        self.save(checkpoint_file_name)

        print('Checkpoint saved')

        self.num_checkpoints += 1

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        self.eval()

class BinaryTreeResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        stream_width,
        output_width,
        linearity=0.99999,
        checkpoint_name='ResRnn',
    ):
        super(BinaryTreeResRnn, self).__init__()

        self.input_width = input_width
        self.stream_width = stream_width
        self.output_width = output_width
        self.linearity = linearity
        self.checkpoint_name = checkpoint_name

        assert(self.input_width <= self.stream_width)
        assert(self.output_width <= self.stream_width)
        assert(self.input_width >= 0)
        assert(self.stream_width >= 0)
        assert(self.output_width >= 0)

        # Variables for coordinating checkpointing
        self.checkpoint_dir = None
        self.num_checkpoints = 0

        self.encl = Res(self.stream_width, self.linearity)
        self.encr = Res(self.stream_width, self.linearity)

        self.decl = Res(self.stream_width, self.linearity)
        self.decr = Res(self.stream_width, self.linearity)

    def _pad_to_stream_width(self, x):
        x_seq_width, x_batch_width, x_element_width = x.size()

        if x_element_width == self.stream_width:
            return x

        stream_width_padding = torch.zeros(
            (
                x_seq_width,
                x_batch_width,
                self.stream_width - x_element_width,
            ),
            device=x.device,
        )
        return torch.cat((x, stream_width_padding), dim=2)

    def _pad_to_even_seq_width(self, x):
        x_seq_width, x_batch_width, x_element_width = x.size()

        if x_seq_width % 2 == 0:
            return x

        seq_width_padding = torch.zeros(
            (1, x_batch_width, x_element_width),
            device=x.device,
        )
        return torch.cat((x, seq_width_padding), dim=0)

    def encode(self, x):
        x_seq_width, x_batch_width, x_element_width = x.size()

        if (x_seq_width, x_element_width) == (1, self.stream_width):
            return x

        x = self._pad_to_stream_width(x)
        x = self._pad_to_even_seq_width(x)

        l = self.encl(x[0::2])
        r = self.encr(x[1::2])

        return self.encode(l + r)

    def decode(self, x, num_steps):
        if num_steps < 0:
            raise ValueError('num_steps must be non-negative')
        if num_steps == 0:
            return x[..., :self.output_width]

        l = self.decl(x)
        r = self.decr(x)

        i = torch.stack([l, r], dim=1).flatten(start_dim=0, end_dim=1)

        return self.decode(i, num_steps - 1)

    def _set_checkpoint_dir_if_none(self, file_name):
        def candidate_paths(path_name):
            yield path_name
            yield from (path_name + '-' + str(i + 2) for i in itertools.count())
        def unique_path(path_name):
            for candidate_path in candidate_paths(path_name):
                if not os.path.exists(candidate_path):
                    return candidate_path

        # Set self.checkpoint_dir if it isn't already
        if self.checkpoint_dir is None:
            abs_save_dir = os.path.dirname(os.path.realpath(__main__.__file__))
            abs_file_name = (
                file_name
                if os.path.isabs(file_name)
                else os.path.join(abs_save_dir, file_name)
            )
            self.checkpoint_dir = unique_path(os.path.dirname(abs_file_name))

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

    def save_checkpoint(self):
        file_name = 'checkpoints/{}/{}.pt'.format(
            self.checkpoint_name,
            self.num_checkpoints
        )

        self._set_checkpoint_dir_if_none(file_name)

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

        print(
            'Saving checkpoint for {} to {}'.format(
                self.checkpoint_name,
                checkpoint_file_name
            )
        )

        # Create dir + checkpoint
        self.save(checkpoint_file_name)

        print('Checkpoint saved')

        self.num_checkpoints += 1

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        self.eval()
