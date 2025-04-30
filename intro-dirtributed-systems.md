---
title: 分布式系统笔记索引（从MIT6.824开始）
date: 2025-04-30 11:24:47
tags:
---

笔者近期因工作需求，正在入手MLSys方向，特别是分布式训练。在追踪前沿技术后，发现自己对分布式系统缺乏系统性了解；于是本篇博客沿着明星课程mit6.824梳理分布式系统脉络，归总学习资料。

课程官网：[https://pdos.csail.mit.edu/6.824/schedule.html](https://pdos.csail.mit.edu/6.824/schedule.html)

MIT6.824中文翻译：[https://mit-public-courses-cn-translatio.gitbook.io/mit6-824](https://mit-public-courses-cn-translatio.gitbook.io/mit6-824)

## 分布式系统概览
构建分布式系统的动力为：
* 通过并行计算，实现**更高计算性能**；
* **容错**机制：在单台服务器故障时，实现向另一台机器的切换；
* 部分系统具有**天然的物理分布**，需要找到**通信协调**方式；
* **分散安全风险**，限制出错域。

分布式**存储系统：KV服务。支持两种操作：put操作会将一个value存入一个key；另一个是get操作会取出key对应的value**。将相同的put请求发送给多个副本，若其中有服务器故障，可能导致服务器之前的不同步。

### CAP理论
不可能三角：
* Consistency（一致性）：Every read receives the most recent write or an error. (强调**数据正确：最新的数据**)
* Availability（可用性）：very request receives a (non-error) response – without the guarantee that it contains the most recent write（强调**不出错：不一定是最新的**）
* Partition tolerance（分区容忍性）：The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes（强调**不挂掉**）

博客：[这可能是我看过最通俗也是最深刻的CAP理论](https://mp.weixin.qq.com/s/6PgqyigrgVICl0JiI73oNg)

#### 一致性
* 强一致：保证get请求可以得到最近一次完成的put请求写入的值。即**保证get到最新数据（一致性）**。如果网络分区或错误而无法保证特定信息是最新的，则系统将返回错误或超时。
    * 实现方案：get/put操作都要询问每一个副本。导致较大通信负担

* 弱一致：**不保证get到最新数据**。**只需要更新最近的数据副本，只需要从最近副本获取数据**。


#### 可用性：处理容错
保证在特定错误类型下，系统依然正常运行，提供完整服务。例如，构建一个有两个拷贝的多副本系统，其中一个故障，另一个还能运行。
* **自我恢复性（Recoverability）**：在出现故障到故障组件被修复期间，系统完全停止工作；但是**修复之后，系统又可以完全正确的重新运行**。一个好的可用的系统，某种程度上应该也是可恢复的。

一些工具：
1. **非易失存储**：硬盘，闪存，SSD之类的。存放一些checkpoint或者系统状态的log在这些存储中，当备用电源恢复，可以从硬盘中读出系统最新的状态，并从那个状态继续运行。
2. **复制**：关键问题是保持副本的同步

#### 可扩展性（Scalability）
我们希望可以通过增加机器的方式来实现扩展，但是现实中这很难实现，需要一些架构设计来将这个可扩展性无限推进下去。

## MapReduce
MapReduce原文：[https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)

MapReduce中文翻译：[https://www.cnblogs.com/fuzhe1989/p/3413457.html](https://www.cnblogs.com/fuzhe1989/p/3413457.html)

Lab1实验：[https://pdos.csail.mit.edu/6.824/labs/lab-mr.html](https://pdos.csail.mit.edu/6.824/labs/lab-mr.html)
