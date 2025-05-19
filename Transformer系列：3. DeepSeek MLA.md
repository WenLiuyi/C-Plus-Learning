## MLA

MLA（Multi-Head Local Attention）的基本思想是：将注意力的输入$h_t$压缩为一个低维的潜在向量，维度为$d_c$，其中$d_c$远小于原始维度$[h, d_{head}]$；在需要计算注意力时，将这个潜在向量映射回高维空间，从而显著减少内存占用。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/1BQYN3xneiaZHmiaxSbwlTR8j5VjxBPiaKgpgg7SUureic8Sw1F7wG75jbj0FicVsl0agJktXhlkibjicwnQ3u7tylibAg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1)

数学公式如下：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/1BQYN3xneiaZHmiaxSbwlTR8j5VjxBPiaKgr7wpzF6TDnYOUEDTicU9PnDDWFeA2v098snBWuffFyxHLFPUDoboYCw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1)

**计算$Q$**：

（37）通过矩阵$W^{DQ}$对$h_t$降维；

（38）通过矩阵$W^{UQ}$对$c_t^Q$升维，$h_t$一降一升，大幅降低了$h_t$本身的权重矩阵参数；

（39）通过矩阵$W^{QR}$对$c_t^Q$进行映射计算，相当于对$c_t^Q$再次降维，再做旋转位置编码；

（40）将$h_t$一降一升后的$q_t^C$，再拼接旋转位置编码$q_t^R$，得到MHA中的$Q$；

**计算$K, V$**：

（41）通过矩阵$W^{DKV}$对$h_t$降维；

（42）通过矩阵$W^{UKV}$对$c_t^{KV}$升维，$h_t$一降一升，也降低了$h_t$本身的权重矩阵参数；

（43）通过矩阵$W^{KR}$对$h_t$进行映射计算，再做RoPE位置编码（与计算$Q$的旋转位置编码不一样）

（44）将$k_t^C$的每个头的计算结果，分别与RoPE位置编码后的$k_t^R$拼接得到$k$，得到MHA中的$K$。

（45）计算$V$矩阵；

**计算Attention再拼接**：

（46）分别计算每个头的注意力；

（47）拼接在一起，利用$W^O$做映射。

上述公式中，只有蓝框处的变量需要被缓存，其他均可利用矩阵吸收重新恢复。

