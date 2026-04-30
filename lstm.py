import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 在 LSTM 中，我们通常把当前输入 x 和上一时刻的隐藏状态 h 拼接(concat)起来一起处理
        concat_size = input_size + hidden_size

        # LSTM 的核心：四个门（用四个线性层来手写，方便你理解它的内部结构）
        
        # 1. 遗忘门 (Forget Gate)：决定丢弃（忘记）多少旧的长期记忆
        self.forget_gate = nn.Linear(concat_size, hidden_size)
        
        # 2. 输入门 (Input Gate)：决定要让多少新的信息进入长期记忆
        self.input_gate = nn.Linear(concat_size, hidden_size)
        
        # 3. 候选记忆 (Candidate Cell)：当前这一个时刻产生的新记忆内容（也叫 c_tilde）
        self.candidate_layer = nn.Linear(concat_size, hidden_size)
        
        # 4. 输出门 (Output Gate)：决定从长期记忆中提取多少信息作为当前的短期记忆（输出）
        self.output_gate = nn.Linear(concat_size, hidden_size)

        # 最终映射到预测结果的输出层
        self.h2y = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        # x: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        if state is None:
            # ！！！重点：LSTM 比 RNN 多了一个状态！
            # h_t 是短期记忆（也就是对外的输出状态）
            # c_t 是长期记忆（Cell State，细胞状态，LSTM 的内部核心履带）
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = state

        outputs = []

        # 时间步循环
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # 将输入与上一步的短期记忆结合起来 [batch_size, input_size + hidden_size]
            combined = torch.cat([x_t, h_t], dim=1)

            # ======= LSTM 内部微操核心公式 =======
            # 1. 遗忘门：Sigmoid 把结果压缩到 0~1（0代表全忘，1代表全留）
            f_t = torch.sigmoid(self.forget_gate(combined))
            
            # 2. 输入门：Sigmoid 决定新的信息哪些放行（0~1）
            i_t = torch.sigmoid(self.input_gate(combined))
            
            # 3. 候选记忆：Tanh 提取出当前要记录的实质性内容（-1~1）
            c_candidate = torch.tanh(self.candidate_layer(combined))
            
            # 4. 输出门：Sigmoid 决定这会儿需要输出啥（0~1）
            o_t = torch.sigmoid(self.output_gate(combined))

            # ======= 更新状态 =======
            # 更新长期记忆 c_t：(旧记忆 * 遗忘门) + (新记忆提议 * 输入门)
            c_t = (f_t * c_t) + (i_t * c_candidate)
            
            # 取出当前长期记忆的一部分，作为对外的短期记忆 / 显示输出
            h_t = o_t * torch.tanh(c_t)

            # 通过短期记忆来做我们实际的业务预测
            y_t = self.h2y(h_t)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, (h_t, c_t)


def make_toy_sequence(seq_len, batch_size):
    # 和 rnn.py 里相同的假数据生成函数
    t = torch.arange(seq_len + 1, dtype=torch.float32)
    series = torch.sin(t * 0.2)

    x = series[:-1].repeat(batch_size, 1).unsqueeze(-1)
    y = series[1:].repeat(batch_size, 1).unsqueeze(-1)
    return x, y


def train_demo():
    torch.manual_seed(42)

    # 替换成了我们手写的 SimpleLSTM
    model = SimpleLSTM(input_size=1, hidden_size=16, output_size=1)
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

    x, y = make_toy_sequence(seq_len=10, batch_size=1)
    preds, _ = model(x)
    
    print("LSTM 测试结果:")
    targets = [round(v, 3) for v in y.squeeze().tolist()]
    outputs = [round(v, 3) for v in preds.squeeze().tolist()]
    print("targets:", targets)
    print("preds  :", outputs)


if __name__ == "__main__":
    train_demo()

# 阀门采用Sigmoid，输出范围在0~1之间，代表信息的通过程度（0完全不通过，1完全通过）
# 候选记忆采用Tanh，输出范围在-1~1之间，代表当前时刻的实质性记忆内容
# 旧记忆与新记忆冲突时就可能为负？