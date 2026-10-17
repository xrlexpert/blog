---
title: VisionEncoder
index_img: /img/banner/vision_encoder.png
date: 2026-07-30 10:12:48
tags:
categories:
---

## Introduction

本文记录若干有关多模态大模型有关视觉编码器相关论文
## Review
### RoPE
RoPE（Rotary Position Embedding）是一种将位置信息注入 Attention 的位置编码方法。

不同于传统位置编码：

$$
x_i'=x_i+p_i
$$

RoPE 不直接修改 token embedding，而是对 Query 和 Key 进行旋转：

$$
q_i'=R(i)q_i
$$

$$
k_i'=R(i)k_i
$$

其中：

- $i$ 表示 token 所处的位置；
- $R(i)$ 表示位置 $i$ 对应的旋转矩阵。


最终 Attention 计算：

$$
Attention(q_i,k_j)
=
(q_i')^Tk_j'
$$

即：

$$
(R(i)q_i)^T(R(j)k_j)
$$

展开：

$$
=q_i^TR(i)^TR(j)k_j
$$

由于旋转矩阵满足：

$$
R(i)^T=R(-i)
$$

因此：

$$
R(i)^TR(j)=R(j-i)
$$


得到：

$$
(q_i')^Tk_j'
=
q_i^TR(j-i)k_j
$$


因此，Attention 分数只与两个 token 的相对位置差：

$$
j-i
$$

有关。


这就是 RoPE 最核心的性质：

> RoPE 不需要显式存储位置 embedding，而是通过旋转 Query 和 Key，使 Attention 计算天然感知 token 间的相对距离。

<details>
<summary>具体实现</summary>
RoPE 的核心操作是：将 hidden dimension 两两分组，在每一个二维子空间中进行旋转。

假设一个 token 的 Query 向量为：

$$
q=[q_0,q_1,q_2,q_3,...,q_{d-1}]
$$

RoPE 首先将其划分为：

$$
(q_0,q_1),(q_2,q_3),...,(q_{d-2},q_{d-1})
$$

每两个维度看作一个二维平面，然后分别进行旋转。


---

### 1. 二维旋转

对于其中任意一组二维向量：

$$
x=
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$


在位置 $m$ 时，对其施加旋转：

$$
x'(m)=R(m\theta_i)x
$$


其中旋转矩阵：

$$
R(m\theta_i)=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}
$$


展开得到：

$$
x'_{2i}
=
x_{2i}\cos(m\theta_i)
-
x_{2i+1}\sin(m\theta_i)
$$


$$
x'_{2i+1}
=
x_{2i}\sin(m\theta_i)
+
x_{2i+1}\cos(m\theta_i)
$$


因此，RoPE 并没有改变向量的信息，而是在二维空间中改变向量的方向。


---

### 2. 不同维度使用不同旋转频率

RoPE 并不是所有维度使用同一个旋转角度。

对于第 $i$ 个二维维度：

$$
\theta_i=\frac{1}{10000^{2i/d}}
$$


因此位置 $m$ 对应的旋转角度为：

$$
\phi_i=m\theta_i
$$


不同维度具有不同的旋转周期：

- 低维度：$\theta_i$ 较大

旋转速度快，可以捕获局部位置变化。


- 高维度：$\theta_i$ 较小

旋转速度慢，可以表示更长距离的位置关系。


这种设计与 Transformer 原始 sinusoidal position encoding 类似：

$$
PE(pos,2i)=sin(pos/10000^{2i/d})
$$

$$
PE(pos,2i+1)=cos(pos/10000^{2i/d})
$$


但区别在于：

传统位置编码：

$$
x_i=e_i+p_i
$$

直接将位置加入 embedding。


而 RoPE 将位置信息注入 Attention 计算过程，即 $q_i=R(i)q$、$k_i=R(i)k$。


---

### 3. RoPE 的矩阵形式

对于 hidden dimension $d=2n$：


RoPE 可以表示为一个 block diagonal 矩阵：

$$
R(m)=
\begin{bmatrix}
R(m\theta_0)&0&...&0\\
0&R(m\theta_1)&...&0\\
...&...&...&...\\
0&0&...&R(m\theta_{n-1})
\end{bmatrix}
$$


其中每一个 block：

$$
R(m\theta_i)=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}
$$


最终 $q_m=R(m)q$、$k_m=R(m)k$，然后送入 Attention：

$$
Attention(q_m,k_n)=q_m^Tk_n
$$


---

### 4. 工程实现中的 rotate_half

实际代码中不会显式构造旋转矩阵 $R(m)$，因为矩阵乘法效率较低。

而是利用二维旋转：

$$
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\rightarrow
\begin{bmatrix}
-x_2\\
x_1
\end{bmatrix}
$$


定义：

$$
rotate\_half(x)=(-x_2,x_1)
$$


因此：

$$
RoPE(x,m)
=
x\cos(m\theta)
+
rotate\_half(x)\sin(m\theta)
$$


对应代码：

```python
q_embed = q * cos + rotate_half(q) * sin
```
</details>

## [Qwen2-VL](https://arxiv.org/pdf/2409.12191)

* 视觉塔由 DFN 的 ViT 初始化，并将其固定位置嵌入替换为轴向 2D-RoPE，因此对输入分辨率不再有尺寸约束，也不需要在换分辨率时插值
* 舍弃固定分辨率输入，改为原生动态分辨率：图像等比例缩放，只要求高宽可被 28 整除，长宽比上限 200
* 视觉 token 采用 2×2 空间合并，每四个合并为一个，并用 `<|vision_start|>` 与 `<|vision_end|>` 界定每个输入的边界
* 引入 M-RoPE，对旋转位置编码做时间、高、宽三个分量的分解，使文本、图像与视频可以共用一套位置语义
* 图与视频采用统一处理方式：图像复制成两帧，视频由相邻两帧构成一个 tube，两者共用同一个视觉编码器与同一条前向
* 视觉塔规模与 LLM 规模解耦，2B / 7B / 72B 共用同一个 675M 视觉塔，视觉侧计算量不随 LLM 放大
* 训练沿用三阶段范式，视觉塔始终挂在 LLM 之后，由语言建模损失驱动
* 阶段三锁定视觉塔，是把前两阶段学到的视觉表示固化下来，使其成为不再改变的交付物

**结构总览** 按数据流可分为四点：

* **规模**：2B / 7B / 72B 三个 LLM 共用同一个 675M 的 ViT，视觉塔规模与 LLM 规模解耦，主干宽度为 `embed_dim = 1280`。
* **图像与视频共用同一前向**：图像先经 `smart_resize` 等比缩放，使高宽均可被 28 整除；patchify 阶段按 2×2 合并块重排像素。图像被复制为两帧相同的 tube，视频则由相邻两帧构成一个 tube。两者均由一个 kernel 为 `[2,14,14]` 的 Conv3d 完成 patch embedding，视觉 token 数正比于 `(T/2) × (H/14) × (W/14)`。
* **ViT 主干**：共 32 层，以轴向 2D-RoPE 提供位置信息（仅含 h / w 两个坐标），以 `cu_seqlens` 实现帧内的变长非因果注意力；PatchMerger 沿序列维将相邻 4 个 token 合并为一个，输出维度 3584 后交给 LLM。
* **LLM 侧**：M-RoPE 为文本、图像、视频统一分配位置，视觉 token 的 h / w 承接视觉塔给出的网格坐标，时间轴则使用独立的时间间隔。

```
pixel_values + grid_thw
  ├─ PatchEmbed: Conv3d(kernel=[2,14,14], stride 相同)     ← 每 2 帧一个 tube，唯一的跨帧算子
  ├─ 按 merge 后网格生成 (h, w) 坐标（block-major，含 temporal 参数但置为 1、不参与）
  ├─ Axial 2D RoPE（只有 h / w 两个坐标，无时间维）
  ├─ Qwen2VLVisionBlock × 32（LayerNorm + 帧内变长非因果注意力 + quick_gelu MLP）
  ├─ PatchMerger: LN → 序列维 concat 相邻 2×2 → Linear → GELU → Linear
  └─ pooler_output → masked_scatter 进 LLM embedding
       └─ LLM 侧 M-RoPE（t / h / w 三路 ID）
```

**训练与初始化** 论文明确给出的部分只有两条。初始化方面，LLM 侧由 Qwen2 的权重初始化，视觉塔由 DFN 的 ViT 初始化，并移除其固定位置嵌入、改为轴向 2D-RoPE（论文原文写作 RoPE-2D）。训练方面，阶段一只训视觉塔，阶段二全参解冻，阶段三锁定视觉塔、只用指令数据精调 LLM。数据规模上，阶段一约 600B token，侧重图文关系、图内文字识别与图像分类；阶段二追加约 800B 图文数据，并按论文所述混入纯文本数据以维持语言能力。三个阶段都只对文本 token 计算监督，视觉 token 不参与损失。

需要区分的是冻结的确切含义：阶段一冻结的是 LLM 的参数更新，而不是计算图。语言损失定义在 LLM 的输出上，梯度必须穿过冻结的 LLM 各层与 merger 才能到达视觉塔，因此视觉塔从始至终都在同一条计算图中由语言损失驱动，不存在脱离 LLM 的独立训练阶段；只是三个阶段的可训练范围不同，冻结 LLM 也并不意味着反传可以省掉。相应地，最终的视觉塔表示由文本监督塑造，而阶段三锁定视觉塔，就是把这份表示固化为不再改变的交付物。

上述关于梯度路径的说明是按论文设计推断的，论文本身只给出了"阶段一只训 ViT"这一句；它也未给出视觉塔与 merger 的学习率、优化器分组、以及三个尺寸是否共用同一份视觉塔初始化，这些需要从官方代码或 checkpoint 比对中确认。

**时间信息如何分配** 需要区分两个层次：交给 LLM 的是绝对时间坐标，而相邻帧之间的信息交互发生在视觉塔内部，且仅限于一次卷积。时间信息共有三条通路：

* **Conv3d 负责相邻帧混合**：`kernel = [temporal_patch_size, patch_size, patch_size] = [2,14,14]`，时间维的卷积核长度与步长相等，因此不产生时序降采样，每个时间步仍输出一个 patch；但该卷积核在时间方向上跨越两帧，所以每个 patch 的特征都取自两帧的对应位置。这是 ViT 中唯一的跨帧算子。
* **视觉塔的注意力不跨帧**：默认配置下 `cu_seqlens` 按帧切分，每段长度为 H·W，共 T 段，帧与帧之间不进行信息交换。因此该视觉塔只在 2 帧的跨度内具备时序能力，并非逐帧图像编码器。
* **视觉 RoPE 不含时间维**：视觉塔调用 `get_vision_position_ids(grid_thw, spatial_merge_size)`，`include_temporal` 取默认值 `False`，位置编码只使用 `(N, 2)` 的 h / w 坐标。视觉塔对帧序不敏感：交换两帧时输出发生变化，但差异只来自卷积核的方向性，而非位置编码。

由此，视觉塔输出的是若干独立帧对构成的表示，其中不包含帧序信息，也不包含帧间关系。LLM 侧的帧序信息仅来自三个离散来源：视觉 token 在文本序列中的排列顺序；M-RoPE 分配的时间坐标（由 `interval = tokens_per_second × temporal_patch_size / fps` 给出，默认 25×2/2 = 50，即每个 tube 递增 50）；以及 Qwen3VL / Qwen3.5 改用 `<3.0 seconds>` 一类文本时间标记后的显式声明。按能力划分：单帧外观与细粒度空间由 ViT 承担；相邻帧的差异只经过一次卷积；跨帧的身份、状态转移与事件顺序没有对应的视觉算子，只能由 LLM 从 token 序列位置与时间坐标中推导。

<div align="center">
  <img src="/img/qwen2vl/qwen2vl_fig2_architecture.jpg" alt="Qwen2-VL 整体架构" style="max-width:100%;">
  <div style="font-size:0.85em;color:#666;">图 2：Qwen2-VL 整体结构，论文原图（arXiv:2409.12191v2）</div>
</div>

配置（`configuration_qwen2_vl.py:27`）：

```python
depth: int = 32
embed_dim: int = 1280          # ViT 主干宽度
hidden_size: int = 3584        # 仅表示 PatchMerger 的输出维度，即 LLM 输入维度
num_heads: int = 16
patch_size: int = 14
spatial_merge_size: int = 2
temporal_patch_size: int = 2
default_rope_type = "axial"
```

**动态分辨率** 论文的核心改动是移除 ViT 的绝对位置嵌入，改用 2D-RoPE。预处理阶段的约束只有一条：缩放后高宽均可被 28 整除，原因是 `factor = patch_size × merge_size = 14 × 2`，2×2 合并要求两个方向的 patch 数均为偶数，否则 merger 中的 `view(-1, 5120)` 会出现形状不匹配。缩放保持等比，不进行裁剪，也不填充为方形，长宽比上限为 200，像素预算限制在 `[56², 28²×1280]`。

```python
# image_processing_qwen2_vl.py:62
def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
```

**Patchify 的块序** 预处理阶段即完成按合并块的像素重排：`reshape(b, c, h//2, 2, p, w//2, 2, p).permute(0, 2, 5, 3, 6, 1, 4, 7)` 得到的序列顺序为 `(h/2, w/2, 2, 2)`，即先遍历合并块、再遍历块内 4 个 patch。由此，序列上相邻的 4 个 token 正是随后被合并的那 4 个，`PatchMerger` 中的 `view(-1, 5120)` 才等价于空间 2×2 池化。单张图像在这一阶段被复制为 `temporal_patch_size = 2` 帧，因此 `image_grid_thw` 的 T 恒为 1（`image_processing_qwen2_vl.py:246`），图像相当于两个完全相同的 tube。

**变长打包** 动态分辨率意味着 batch 内各样本长度不同。实现不采用 padding，而是将所有 patch 拼接为一条序列，由 `cu_seqlens` 切分出 block-diagonal 的注意力段：

```python
# vision_utils.py:64
seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)
```

`merge_temporal` 默认为 `False`，即每帧为独立片段（H·W 一段，共 T 段），说明 ViT 内部不进行跨帧建模，输出是帧内最优的表示，时序关系完全交由 LLM 侧的位置编码处理。`VisionAttention` 的序列维为第 0 维，不存在真实 batch 维（以 `unsqueeze(0)` 构造虚拟维）；且仅当请求 FA2 时才传入 `cu_seqlens`，其他后端会退化为按 `cu_seqlens` 逐段切分的循环，功能等价但开销不同。

**轴向 2D-RoPE** `head_dim = 1280/16 = 80`，频率表按 `head_dim//4` 计算得 20 个频率，H 和 W 各用 20 个，拼接成 40 后再复制一份补到 80，正好匹配 `rotate_half` 的约定：

```python
# modeling_qwen2_vl.py:228
class Qwen2VLVisionRotaryEmbedding(nn.Module):
    """Simple axial 2D rope with same freqs used for H and W grids."""

    def __init__(self, config: Qwen2VLVisionConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = self.compute_axial_rope_parameters(self.config, device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.embed_dim // config.num_attention_heads
        spatial_dim = dim // 2                                  # 80 // 2 = 40
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), 1.0                          # attention_factor 在本类 RoPE 中不使用

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (2, N) — row 0 = h coords, row 1 = w coords
        position_ids_expanded = position_ids[..., None].float()
        freqs = position_ids_expanded * self.inv_freq.float()
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        return self.recomposition_frequencies(cos), self.recomposition_frequencies(sin)

    def recomposition_frequencies(self, freq):
        freq_h, freq_w = freq[:, 0], freq[:, 1]                  # 各 20 个频率
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)            # 20 + 20 = 40
        return torch.cat([freq_hw, freq_hw], dim=-1)             # 复制一份 → 80，对齐 rotate_half
```

模型不存在 `pos_embed` 参数。绝对位置表与输入尺寸绑定，改变分辨率必须插值，而插值会破坏频率一致性；轴向 RoPE 的每个 token 由 (h, w) 坐标直接计算，任意分辨率都落在同一套频率上，代价是位置信息中不再包含学习得到的全局布局先验。

**位置坐标必须与块序对齐** 坐标生成位于 `vision_utils.py:81`，以 `reshape(h//m, m, w//m, m).transpose(1, 2)` 将坐标排为 `(块行, 块列, 块内行, 块内列)`，与 patchify 的 permute 严格互逆。该约束在修改 ViT 时容易被破坏：任何在早期进行 token 选择、丢弃或重排的方案（codec top-k、动态压缩、帧选择）都必须按块序语义同步重编号位置，否则保留 token 之间的相对几何关系会出现偏差；此类偏差在图像基准上几乎不可见，只在局部空间或时序关系任务上暴露。

**PatchEmbed 与 PatchMerger** patch embedding 用 3D 卷积，`kernel = [2, 14, 14]` 且 stride 相同：

```python
# modeling_qwen2_vl.py:294
class PatchEmbed(nn.Module):
    def __init__(
        self, patch_size: int = 14, temporal_patch_size: int = 2,
        in_channels: int = 3, embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]          # [2, 14, 14]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```

token 数因此正比于 `(T/2) × (H/14) × (W/14)`，而非 `T × H × W`，长视频的序列长度相应减半，这也是图像与视频能够共用同一套预处理与前向的原因。压缩在序列维完成，而非空间维：

```python
# modeling_qwen2_vl.py:320
class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)             # 1280 * 4 = 5120
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),                   # 5120 -> 5120
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),                                # 5120 -> 3584 (LLM hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # view 把序列维上相邻的 4 个 token 拼成一个 5120 维向量，等价于空间 2x2 池化
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x
```

ViT 主干采用 LayerNorm + quick_gelu（DFN/CLIP 血统），LLM 采用 RMSNorm + silu，两套归一化在同一模型内并存，因此 ViT 侧的改动不会被 LLM 的归一化补偿。此外，`Qwen2VisionTransformerPretrainedModel.forward` 返回的 `last_hidden_state` 是 merger 之前的 patch 特征，`pooler_output` 才是交给 LLM 的视觉 token，`get_image_features` 取用的是后者。

**token 预算** 论文中的示例可直接复算：224×224 的图像对应 16×16 = 256 个 patch，经 2×2 合并后得到 64 个视觉 token，加上 `<|vision_start|>` 与 `<|vision_end|>` 共 66 个。

**M-RoPE** 该模块位于 LLM 侧，负责为视觉 token 分配坐标。head 维按 `mrope_section = [16, 24, 24]` 切为三段，再按 `i % 3` 交错取用，最后复制一份补足完整 head 维：

```python
# modeling_qwen2_vl.py:171，Qwen2VLRotaryEmbedding.recomposition_frequencies
def recomposition_frequencies(self, freq):
    """Recompose the frequencies into the final spatial layout used per each grid."""
    freq = torch.cat([m[i % 3] for i, m in enumerate(freq.split(self.mrope_section, dim=-1))], dim=-1)
    return torch.cat((freq, freq), dim=-1)
```

文本输入时三行位置 ID 相同，M-RoPE 退化为标准 1D-RoPE，这是它不损伤文本能力的原因。图像与视频则将时间、高、宽拆分为三路 ID，跨模态接续时取"上一段最大坐标 + 1"：

```python
# modeling_qwen2_vl.py:1013，Qwen2VLModel.get_rope_index 的模态分段循环（节选）
for modality_type, start_idx, end_idx in input_type_group:
    # text == 0：三路 ID 共用同一套递增坐标，退化为 1D-RoPE
    if modality_type == 0:
        text_len = end_idx - start_idx
        llm_pos_ids_list.append(
            torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
        )
        current_pos += text_len
    # image == 1, video == 2：按该模态的 (t, h, w) 网格生成三路坐标
    else:
        grid_thw = next(grid_iters[modality_type])
        vision_position_ids = self.get_vision_position_ids(
            current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
        )
        llm_pos_ids_list.append(vision_position_ids)
        # 下一个模态的起点 = 本段最大坐标 + 1；h/w 用合并后的网格
        current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
```

h / w 采用合并后的网格，因此 224² 图像的坐标范围仅为 0..7；取 `max` 是因为高瘦或扁宽图像的两轴最大值不同，需要保证坐标不重叠。由此视觉 token 占用的 ID 范围远小于序列长度，模型可以在推理阶段外推到更长的序列。时间轴上的间距由 `interval = tokens_per_second × temporal_patch_size / fps` 给出（docstring 示例：fps=1、25 token/s、temporal_patch_size=2 时步长为 50）。

<div align="center">
  <img src="/img/qwen2vl/qwen2vl_fig3_mrope.png" alt="M-RoPE 示意" style="max-width:100%;">
  <div style="font-size:0.85em;color:#666;">图 3：M-RoPE 将旋转位置编码分解为时间、高、宽三个分量，论文原图（arXiv:2409.12191v2）</div>
</div>

生成阶段使用缓存的 `rope_deltas` 对新 token 的坐标进行平移，并与 prefill 阶段保持连续；打包输入时约定：当 `position_ids` 为 `[4, bs, seq]` 时，第 0 行为纯文本位置，用于构造 mask，后 3 行才是 M-RoPE。

**Qwen2.5-VL 的改动** 视觉塔共四处变更：归一化由 LayerNorm 改为 RMSNorm（`qwen2_5_vl/modeling_qwen2_5_vl.py:349`）；注意力改为以窗口注意力为主、少数层使用全注意力（`fullatt_block_indexes`）；token 顺序按窗口重排，计算完成后再还原；merger 的归一化同样改为 RMSNorm。重排以 `spatial_merge_unit = spatial_merge_size²` 为单位：

```python
# qwen2_5_vl/modeling_qwen2_5_vl.py:427
input_tensor = input_tensor.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
input_tensor = input_tensor[window_index, :, :]
input_tensor = input_tensor.reshape(seq_len, -1)
```

位置嵌入需要执行相同的重排，否则 RoPE 坐标与 token 不对齐。同时需要注意，merger 在还原之前执行：

```python
# qwen2_5_vl/modeling_qwen2_5_vl.py:487
reverse_indices = torch.argsort(window_index)
merged_hidden_states = self.merger(hidden_states)
merged_hidden_states = merged_hidden_states[reverse_indices, :]
```

**图与视频统一** 视频帧按 fps 采样，帧数强制为 `temporal_patch_size` 的倍数，每帧的像素上限由总帧数反推，从而将整段视频的 token 量限制在预算之内：

```python
# video_processing_qwen2_vl.py:248
min(max_pixels, total_pixels * temporal_factor // num_frames)
```

**小结** Qwen2-VL 将分辨率适应性从"预处理配合位置插值"转移为"任意长度打包、逐 token 生成的 2D-RoPE，以及固定比例的 2×2 合并"，因此不需要任何与尺寸绑定的参数或缓存。代价是必须全程携带 `grid_thw`，并在 ViT 与 SDPA 路径中手写变长分支。

<details>
<summary>实现细节</summary>

位置坐标生成（`vision_utils.py:81`）：

```python
def get_vision_position_ids(grid_thw, spatial_merge_size, include_temporal=False, kwargs=None):
    """...position_ids: (total_tokens, 2)，spatial indices laid out block-major over m×m blocks."""
    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)
```

注意力（`modeling_qwen2_vl.py:385`）：

```python
class VisionAttention(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1                       # eager attention 需要
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False                              # 视觉塔非因果

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
        seq_length = hidden_states.shape[0]                 # 序列维是第 0 维，没有真实 batch 维
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)   # 构造虚拟 batch 维
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        if is_flash_attention_requested(self.config):
            # FA2 可原生消费 cu_seqlens，一次完成变长注意力
            attn_output, _ = attention_interface(
                self, query_states, key_states, value_states, attention_mask=None,
                scaling=self.scaling, dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen, max_length_k=max_seqlen, is_causal=False, **kwargs,
            )
        else:
            # 其他后端按 cu_seqlens 逐段切开分别计算，再拼回
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=2) for t in (query_states, key_states, value_states)]
            attn_outputs = [attention_interface(self, q, k, v, attention_mask=None, is_causal=False)[0]
                            for q, k, v in zip(*splits)]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)
```

融合进 LLM（`modeling_qwen2_vl.py:1179`）：

```python
    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # 视觉塔输出 → 按 split_sizes 切回每张图 → 拼成一条序列
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            # 占位符位置掩码：input_ids == image_token_id，并校验数量一致
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            # 把视觉 embedding 就地写入占位符位置
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(input_ids=None, inputs_embeds=inputs_embeds, ...)
        return Qwen2VLModelOutputWithPast(last_hidden_state=outputs.last_hidden_state, ...)
```

`get_image_features` 中的切分长度为 `image_grid_thw.prod(-1) // spatial_merge_size**2`，即 `T·H·W / 4`，与 PatchMerger 的压缩比严格对应。`get_placeholder_mask` 会校验 `n_image_tokens × hidden == image_features.numel()`，数量不匹配直接报错，因此视觉塔输出的 token 数必须与输入文本中的占位符数量一致。特殊 token 定义在配置中：`image_token_id=151655`、`video_token_id=151656`、`vision_start_token_id=151652`、`vision_end_token_id=151653`。

</details>


## [Qwen3VL](https://arxiv.org/pdf/2511.21631)

* 模型架构采用SigLip2（SigLIP2-SO-400M用于4B以上的大模型，SigLIP2-Large 300M用于4B及以下的模型）
* 舍弃了Qwen2.5VL中MRoPE 多模态旋转编码的形式，采用绝对时间文本token如 `<3.0 seconds>`的形式显示处理视频, 将时序能力交给llm处理
* 原生多模态训练

## [Qwen3.5](https://arxiv.org/pdf/2511.21631)

* 模型架构采用SigLip2（SigLIP2-SO-400M用于4B以上的大模型，SigLIP2-Large 300M用于4B及以下的模型）
* 舍弃了Qwen2.5VL中MRoPE 多模态旋转编码的形式，采用绝对时间文本token如 `<3.0 seconds>`的形式显示处理视频, 将时序能力交给llm处理
* 原生多模态训练

## [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/DeepSeek_VL2_paper.pdf)



## [KiMi 2.5](https://arxiv.org/pdf/2602.02276)

MoonViT-3D:

## [Kimi3](https://arxiv.org/pdf/2607.24653)

MoonVit: 2d可学习空间位置编码, Siglip2初始化。

MoonVitV2（Kimi K3）：3d位置编码，2d可学习空间位置编码 + 1d 固定时间位置编码，无Siglip2初始化。 Next-Token Prediction原生多模态训练

## LLaVA-OneVision 2



## JoyAI-VL-Interaction



## Mage-VL

Mage-ViT：Codec原生，采用ov2相同的预训练方式



Stage1：接入LLM时候是一起训练的，并没有区分S1 projector , S2 全参训练

* 350M图像caption数据 + 4.2M短视频caption数据



短视频 SFT

详细视频描述数据

## CoVisco

## AdaSpark