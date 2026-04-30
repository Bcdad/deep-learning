import torch
from torch import nn
import math

#d_model 是 Transformer 内部的一个重要超参数，代表了每个词被映射到的抽象特征空间的维度大小。它决定了模型的表达能力和计算复杂度。
# 1. 词嵌入与位置编码 (让模型不仅能看懂词，还能知道词排在第几个)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # === 原版：正弦余弦公式生成的绝对位置编码 (固定不训练) ===
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe.unsqueeze(0)) # 独立于参数外，不会被优化器更新

        # === 【如需训练】把它改成可学习的位置编码 (Learned Positional Encoding)，如 BERT 所用 ===
        # 核心思想：不用公式算了。我们直接声明一个和“词典库”一模一样的“位置库”。
        # 相当于把“第1号位”、“第2号位”也当成独立的词，让模型在反向传播中自己去微调每一个位置应该长什么样的向量。
        self.pe_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x 的形状是 [batch_size, seq_len, d_model]
        
        # === 原版固定拼接 ===
        # return x + self.pe[:, :x.size(1), :]

        # === 【如需训练】前向传播里的改法 ===
        seq_len = x.size(1)
        # 1. 生成一个表示当前序列每个位置索引的列表：[0, 1, 2, ..., seq_len-1]
        #    需要保证和输入 x 在同一个设备(CPU/GPU)上
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0) # 形状: [1, seq_len]
        
        # 2. 拿着索引去查表，提取出当前被不断训练更新着的位置特征向量
        learned_pe = self.pe_embedding(positions) # 形状: [1, seq_len, d_model]
        
        # 3. 同样是加法：把词本身的特征 + 此时此地的位置特征 融合在一起
        return x + learned_pe


# 2. 注意力机制的核心：纯手工单头自注意力 (Single-Head Attention)
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 用 3 个极其普通的线性层，分别扮演 3 个不同的职能：Q, K, V
        self.q_linear = nn.Linear(d_model, d_model) # Query: 我带着找什么样的信息？
        self.k_linear = nn.Linear(d_model, d_model) # Key: 我自己拥有什么样的特征属性？
        self.v_linear = nn.Linear(d_model, d_model) # Value: 老实交出我的实际内容。
        self.d_model = d_model

    def forward(self, q_input, k_input, v_input, mask=None):
        Q = self.q_linear(q_input) # [batch, seq_len_q, d_model]
        K = self.k_linear(k_input) # [batch, seq_len_k, d_model]
        V = self.v_linear(v_input) # [batch, seq_len_v, d_model] (和 K 一样长)

        # 核心公式 1: Q 乘以 K的转置 (打分) -> 看看谁和谁最匹配
        # 注意：K.transpose(-2, -1) 仅仅是把最后的两个维度矩阵对调一下来算乘法
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            # 如果有遮罩，比如解码器在预测第N个词时绝对不能偷看未来第N+1个词的答案，
            # 就把那些不被允许看的格子的分数强制设为极小值（比如 -1e9）
            scores = scores.masked_fill(mask == 0, -1e9)

        # 核心公式 2: Softmax 把这些相互的打分，变成严格加起来等于1的百分比概率权重 (0~1)
        attn_weights = torch.softmax(scores, dim=-1)

        # 核心公式 3: 用刚才算出来的关注权重，去按比例乘以实际的物品 V，把注意力凝聚成最后合并的结果
        output = torch.matmul(attn_weights, V)
        return output


# 3. 编码器层 (Encoder Layer): 把一段话转换成高维且带有上下文语境的深层理解
class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 思考加工车间 (前馈神经网络)，在提取到各方的注意力后，自己独立咀嚼一遍
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # 步骤 1: 自注意力 (对于编码器而言，是自己看自己内部互相联系，所以 Q,K,V 都是传入 x)
        attn_out = self.attention(x, x, x, mask)
        # 残差连接 (Add) + 归一化 (Norm)：这是 Transformer 训练不崩的法宝
        x = self.norm1(x + attn_out)
        
        # 步骤 2: 独立深入思考 (FFN)，每个词在自己的位置上深度加工
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# 4. 解码器层 (Decoder Layer): 一边观察着编码器的成果，一边一点点生成下文
class DecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attention = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 这个比 Encoder 多出来的核心部件：交叉注意力层！负责连接“两套大世界”
        self.cross_attention = SimpleAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, trg, memory, trg_mask, src_mask):
        # 步骤 1: 遮蔽自注意力 (只能看前面自己已经生成的词，不能偷看后文)
        _trg = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + _trg)
        
        # 步骤 2: === 极其关键的交叉注意力 (Cross-Attention) ===
        # Q 是解码器自己现在的半成品状态; K 和 V 来自刚才外接的编码器整理好的标准答案全集 (memory)
        _trg = self.cross_attention(trg, memory, memory, src_mask)
        trg = self.norm2(trg + _trg)
        
        # 步骤 3: 独立思考
        _trg = self.ffn(trg)
        trg = self.norm3(trg + _trg)
        return trg


# 5. 完整的 Transformer 拼装厂
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 字典映射：把诸如数字'5'或文字'cat'转换为长度 d_model 的抽象向量空间位置
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 装配单层编码器和解码器 (真实的大模型如 ChatGPT 会装这玩意几十上百层)
        self.encoder = EncoderLayer(d_model)
        self.decoder = DecoderLayer(d_model)
        
        # 把最终高维理解，重新翻译回字典词表的分类概率，看看到底应该选哪个词作为下个字
        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        # 创建遮罩，为了批量运行我们有时会在结尾补 0 (PAD)
        # 掩盖掉所有的 <PAD> 让编码器不要把注意力浪费在无意义的空白格子上
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        # 1. 也是先掩盖掉 <PAD>
        pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)
        # 2. 掩盖掉未来单词 (生成一个下三角形状为 True，别的为 False 的掩码网格)
        trg_len = trg.shape[1]
        subsequent_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        # 两者相综合
        return pad_mask & subsequent_mask

    def forward(self, src, trg):
        # src: 待翻译的原话 [batch, src_len]
        # trg: 我们拿来供其一步步下棋做生成的参考前文 [batch, trg_len]
        
        # 第一步：制作掩码 (Mask) —— 保证模型不作弊，并无视垃圾填充符号(PAD)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 第二步：词潜入与位置注入，让它们拥有身份地位
        src_emb = self.pos_encoding(self.embedding(src))
        trg_emb = self.pos_encoding(self.embedding(trg))

        # 第三步：编码器疯狂压缩提炼信息 -> 产出高级结果 Memory
        memory = self.encoder(src_emb, src_mask)
        
        # 第四步：解码器按照 Memory 以及目前的半句话进程，推导下一步全盘
        out = self.decoder(trg_emb, memory, trg_mask, src_mask)
        
        # 第五步：翻字典，得出最终的预测概率打分
        logits = self.fc_out(out)
        return logits


# === 下面是供测试验证的一个小任务试验 ===
def train_demo():
    torch.manual_seed(42)
    # 这次我们不做纯粹正弦波了，既然是机器翻译模型，那就玩个词汇序列游戏！
    # 任务：“源端输入几个胡乱摆的数字系列，目标是反向输出它”。 
    # 比如输入 [3, 4, 5, 6]，模型得吐出 [6, 5, 4, 3] 就算学懂了真正的逻辑映射。
    
    # 词汇表(Vocab): 0代表PAD，1代表SOS(开始标志)，2代表EOS(结束标志)，3-9是实际数据
    vocab_size = 10
    d_model = 16
    
    model = SimpleTransformer(vocab_size, d_model)
    # 因为任务更复杂，我们把学习率相对调下一点
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Transformer 最配的不是 MSE，因为它是“选词”，所以用分类交叉熵 CrossEntropy，还要忽略所有的填充0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # 我们制造一个固定的玩具数据: [源序列, 输入解码端参考线索, 终极目标大答案]
    # 对解码器而言，它的第一步必须吃一个固定的SOS头(1)，用来猜正文的第一个数据(6)
    src_data = torch.tensor([[3, 4, 5, 6, 0]])    # 原题，尾部有个PAD(0)
    trg_in = torch.tensor([[1, 6, 5, 4, 3]])      # 1=SOS。解码阶段分别看到 [1], 然后 [1,6], 然后 [1,6,5]
    trg_out = torch.tensor([[6, 5, 4, 3, 2]])     # 模型对应上面的输入应当预测 [6], 然后[5], 最后是个[EOS(2)]

    print("开始训练极其初级的序列反转Transformer...")
    for epoch in range(1, 101):
        logits = model(src_data, trg_in)  # 输出形状 [1, 5, vocab_size]
        
        # 因为分类标准的函数只能吃2维的对照表，所以将 3 维直接展平了比对答案
        logits = logits.view(-1, vocab_size) 
        target = trg_out.view(-1)
        
        loss = loss_fn(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    print("\n训练结束！来看下成果评估！")
    # 我们不教它答案了，就在最后的权值下让他做刚才的题。
    # argmax(dim=-1) 是从预测的词表概率中挑出那个最笃定的号码
    preds = model(src_data, trg_in).argmax(dim=-1)
    
    print(f"标准原题: [ 3, 4, 5, 6 ]")
    print(f"模型作答: {preds[0].tolist()} (对应标志: 1=SOS, 2=EOS, 0=PAD)")

if __name__ == "__main__":
    train_demo()
