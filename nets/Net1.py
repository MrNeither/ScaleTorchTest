import torch


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv1d(22, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.c3 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        self.f1 = torch.nn.Linear(16 * 4, 32)
        self.relu = torch.nn.ReLU()
        self.f2 = torch.nn.Linear(32, 3)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.flatten(start_dim=1)
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        return x
