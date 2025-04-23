RL和RLHF区别：https://blog.51cto.com/u_16163452/13046401
RL：持续交互，探索多个状态，最大化多步累积奖励
RLHF：单步文本生成+静态奖励模型（SFT，训练奖励模型，策略优化PPO调整LLM参数），一次性评分
    LLM：最大化下一个token似然概率

RLHF：基于人类反馈的强化学习框架
Actor，Critic--动态更新
Reward，Reference-参数冻结
loss体系：critic+reward+reference；综合它们的结果计算loss，用于更新Actor和Critic Model

Actor：输入一条prompt，得到一条response，prompt+response送入loss体系，计算得loss，用于更新actor（SFT初始化）每一个token有它对应的log_prob
reference：输入actor产生的prompt+response，得到ref_log_probs
    偏离程度：ref_log_probs-log_probs（KL散度）

Critic：预测总收益Vt（deepspeed-chat的实现保持一致：从RW阶段的Reward Model初始化而来）
Reward：
    1. reward为什么冻结？已经过估算收益的训练，产生客观值；代表即时收益，token已产生，收益立刻算出
    2. 为什么有了critic，还要reward：上帝视角Vt不知道，使用Rt+\gamma*V(t+1)近似

    RLHF中，response的reward只用最后一个token。即：prmopt+response只让reward model推理一次，作为整个response的reward。
    每个token的reward：当前模型和参考模型的KL散度


1. Advantage：实际收益-预测收益：Rt+\gamma*V(t+1)-Vt+\gamma * \lambda * Adv(t+1)  GAE

2. actor Loss：对St而言，如果token At产生的收益较高，那就增大它出现的概率，否则降低它出现的概率。

3. critic loss：


流程：
1. 准备一个batch的prompts；
2. 将这个batch的prompts输入actor，得到response；
3. prompt+responses输入给 Critic/Reward/Reference，分别计算得得到所有 token 的 values、最后一个 token 的 reward 和所有 token 的 log probs，按照强化学习的术语，称这些数据为经验（experiences）了；
4. 根据 experiences 多轮计算 actor loss 和 critic loss ，并更新 Actor 和 Critic 模型。

对于第 4 步，我们当然可以一轮 experiences 就更新一次 actor 和 critic，但是为了尽可能利用这个 batch 的 experiences，我们对 actor 和 critic 做多轮更新。我们将 experiences 中多轮更新开始前的 log probs 和 values 称为 old log probs 和 old values（reward 不会多轮计算）。在每一轮中，actor 和 critic 会生成 new log probs 和 new values，然后在 old 的基础上计算 actor loss 和 critic loss，然后更新参数。



## FSDP
[Pytorch官方文档](https://pytorch.ac.cn/tutorials/intermediate/FSDP_tutorial.html)