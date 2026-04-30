import torch
from torch import nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 使用最基础的线性层手写 RNN Cell，便于理解公式
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # 将每个时间步的隐藏状态映射到输出
        self.h2y = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        # x: [batch, seq_len, input_size] [批次大小, 序列长度, 特征维度]
        # x0: [batch_size, input_size] 常见
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            # 初始隐藏状态全零
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            # 允许外部传入初始隐藏状态
            h_t = h0

        outputs = []

        # 按时间步展开循环，体现 RNN 的递归结构
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
            y_t = self.h2y(h_t)
            outputs.append(y_t)

        # torch.stack 的作用是把列表里分散的张量组合成一个大张量，并在这个过程中插入一个新的维度。
        # outputs 列表中有 seq_len 个二维张量，每个张量形状是 [batch_size, output_size] (代表每一个单步的预测)。
        # dim=1 表示在索引为 1 的位置（即中间的时间维度）进行堆叠“组装”。
        # 最终拼接每个时间步的输出，形状恢复为三维结构：[batch, seq_len, output_size]
        y = torch.stack(outputs, dim=1)
        return y, h_t


def make_toy_sequence(seq_len, batch_size):
    # 生成简单的正弦序列，目标是预测下一个时间步的值
    t = torch.arange(seq_len + 1, dtype=torch.float32)
    series = torch.sin(t * 0.2)

    x = series[:-1].repeat(batch_size, 1).unsqueeze(-1)
    y = series[1:].repeat(batch_size, 1).unsqueeze(-1)
    return x, y


def train_demo():
    torch.manual_seed(42)

    model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    epochs = 300
    seq_len = 25
    batch_size = 32

    for epoch in range(1, epochs + 1):
        x, y = make_toy_sequence(seq_len, batch_size)
        preds, _ = model(x)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    # 简单预测：取一个样本查看拟合效果
    x, y = make_toy_sequence(seq_len=10, batch_size=1)
    preds, _ = model(x)
    targets = [round(v, 3) for v in y.squeeze().tolist()]
    outputs = [round(v, 3) for v in preds.squeeze().tolist()]
    print("targets:", targets)
    print("preds  :", outputs)


if __name__ == "__main__":
    train_demo()
