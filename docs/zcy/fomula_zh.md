# 面向实时 GNSS/INS 组合导航的学习辅助 GNSS 扰动状态模型

## 1. 问题定义

本文保持现有 `KF-GINS` 误差状态扩展卡尔曼滤波器作为主干，仅在 GNSS 量测更新侧引入一个轻量学习模块，用于估计城市退化环境下的 GNSS 未建模扰动。

目标场景为：

- 实时 GNSS/INS 组合导航
- 城市峡谷、遮挡、多路径等退化环境
- GNSS 并未完全中断，但量测中存在显著系统偏差

本文方法的核心思想为：

- 不替代现有物理滤波主干
- 不直接使用神经网络输出位置结果
- 不仅仅对量测协方差进行自适应调整
- 而是显式学习一个低维 GNSS 扰动状态，并将其结构化嵌入 ESKF 的 GNSS 更新过程

## 2. 名义状态定义

沿用 KF-GINS 中的名义状态定义，可表示为

\[
\mathbf{x}^{n} =
\begin{bmatrix}
\mathbf{p} \\
\mathbf{v} \\
\mathbf{R}_{bn} \\
\mathbf{b}_g \\
\mathbf{b}_a \\
\mathbf{s}_g \\
\mathbf{s}_a
\end{bmatrix}
\]

其中：

- \(\mathbf{p} \in \mathbb{R}^3\)：位置
- \(\mathbf{v} \in \mathbb{R}^3\)：速度
- \(\mathbf{R}_{bn} \in SO(3)\)：姿态旋转矩阵
- \(\mathbf{b}_g, \mathbf{b}_a \in \mathbb{R}^3\)：陀螺与加速度计零偏
- \(\mathbf{s}_g, \mathbf{s}_a \in \mathbb{R}^3\)：陀螺与加速度计比例因子误差

## 3. 误差状态模型

误差状态向量定义为

\[
\delta \mathbf{x} =
\begin{bmatrix}
\delta \mathbf{p} \\
\delta \mathbf{v} \\
\delta \boldsymbol{\phi} \\
\delta \mathbf{b}_g \\
\delta \mathbf{b}_a \\
\delta \mathbf{s}_g \\
\delta \mathbf{s}_a
\end{bmatrix}
\in \mathbb{R}^{21}
\]

其中：

- \(\delta \mathbf{p}\)：位置误差
- \(\delta \mathbf{v}\)：速度误差
- \(\delta \boldsymbol{\phi}\)：姿态误差
- \(\delta \mathbf{b}_g, \delta \mathbf{b}_a\)：IMU 零偏误差
- \(\delta \mathbf{s}_g, \delta \mathbf{s}_a\)：IMU 比例因子误差

系统传播过程保持 KF-GINS 原始形式不变：

\[
\delta \mathbf{x}_{k+1} = \mathbf{\Phi}_k \, \delta \mathbf{x}_k + \mathbf{w}_k
\]

\[
\mathbf{P}_{k+1} = \mathbf{\Phi}_k \mathbf{P}_k \mathbf{\Phi}_k^\top + \mathbf{Q}_k
\]

其中：

- \(\mathbf{\Phi}_k\) 为状态转移矩阵
- \(\mathbf{Q}_k\) 为离散过程噪声协方差
- \(\mathbf{P}_k\) 为误差状态协方差矩阵

本文不对惯导机械编排与状态传播部分进行修改，仅在 GNSS 量测更新时引入学习扰动项。

## 4. 原始 GNSS 量测更新模型

设 GNSS 在导航坐标系下的位置量测为

\[
\mathbf{z}_k^{g} \in \mathbb{R}^3
\]

名义状态下预测得到的 GNSS 天线相位中心位置为

\[
\hat{\mathbf{p}}_{a,k}
\]

则 KF-GINS 中 GNSS 位置更新所使用的创新量可写为

\[
\mathbf{d}_k = \hat{\mathbf{p}}_{a,k} - \mathbf{z}_k^{g}
\]

更一般地，也可写为非线性量测模型下的残差形式：

\[
\mathbf{r}_k = \mathbf{z}_k^{g} - h(\hat{\mathbf{x}}_k)
\]

其中 \(h(\cdot)\) 表示名义状态到 GNSS 位置量测空间的映射。

对该量测模型进行线性化，可得量测雅可比矩阵

\[
\mathbf{H}_k = \frac{\partial h}{\partial \delta \mathbf{x}}\Big|_{\hat{\mathbf{x}}_k}
\]

设 GNSS 量测噪声满足

\[
\mathbf{n}_k \sim \mathcal{N}(0,\mathbf{R}_k)
\]

则标准 ESKF 更新为

\[
\mathbf{K}_k = \mathbf{P}_k \mathbf{H}_k^\top \left( \mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^\top + \mathbf{R}_k \right)^{-1}
\]

$$
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-} + \mathbf{K}_k \left( \mathbf{r}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-} \right)
$$

\[
\mathbf{P}_k^{+} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)\mathbf{P}_k^{-}(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top
\]

## 5. 本文引入的 GNSS 扰动状态

为刻画城市退化环境下 GNSS 量测中的未建模系统偏差，本文引入一个低维 GNSS 扰动状态：

\[
\mathbf{d}_k^{b} \in \mathbb{R}^3
\]

该扰动状态定义在导航坐标系下，其物理含义为：

- 由多路径、遮挡、非视距传播等引起的 GNSS 位置量测偏差
- 随时间变化的、场景相关的未建模扰动项

需要强调的是，\(\mathbf{d}_k^{b}\) 并不是：

- 完整的 GNSS 误差模型
- 量测协方差缩放因子
- 直接替代导航解的伪位置输出

它仅作为 GNSS 量测更新中的一个结构化补偿项存在。

## 6. 引入扰动状态后的量测模型

将 GNSS 退化量测建模为

\[
\mathbf{z}_k^{g} = h(\mathbf{x}_k) + \mathbf{d}_k^{b} + \mathbf{n}_k
\]

其中：

- \(\mathbf{d}_k^{b}\) 为待学习的 GNSS 扰动项
- \(\mathbf{n}_k\) 为零均值高斯噪声

则校正后的创新量可表示为

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

或在 KF-GINS 当前实现语义下写成

\[
\tilde{\mathbf{d}}_k = \mathbf{d}_k - \hat{\mathbf{d}}_k^{b}
\]

随后利用校正后的创新进行标准 ESKF 更新：

\[
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-}
+ \mathbf{K}_k \left( \tilde{\mathbf{r}}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-} \right)
\]

这也是本文与传统自适应噪声方法的本质区别：

- 传统方法通过调整 \(\mathbf{R}_k\) 改变量测置信度
- 本文方法显式估计并补偿量测中的结构化偏差项

## 7. 学习模型定义

定义一个轻量神经网络

\[
f_{\theta}(\cdot)
\]

用于输出 GNSS 扰动状态估计值

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

其中 \(\mathbf{u}_k\) 为在 GNSS 更新时刻构造的实时特征向量。

## 8. 实时输入特征设计

为保证方法具有实时可部署性，网络输入仅使用在线可获得的等价特征。

第一版建议输入定义为

\[
\mathbf{u}_k =
\begin{bmatrix}
\mathbf{r}_k \\
\boldsymbol{\sigma}_k^{g} \\
\mathbf{v}_k \\
\bar{\boldsymbol{\omega}}_k \\
\bar{\mathbf{a}}_k \\
\mathbf{r}_{k-1} \\
\mathbf{r}_{k-2}
\end{bmatrix}
\]

其中：

- \(\mathbf{r}_k \in \mathbb{R}^3\)：当前 GNSS 创新
- \(\boldsymbol{\sigma}_k^{g} \in \mathbb{R}^3\)：GNSS 位置标准差
- \(\mathbf{v}_k \in \mathbb{R}^3\)：当前速度估计
- \(\bar{\boldsymbol{\omega}}_k\)：短时间窗 IMU 角速度统计量
- \(\bar{\mathbf{a}}_k\)：短时间窗 IMU 加速度统计量
- \(\mathbf{r}_{k-1}, \mathbf{r}_{k-2}\)：最近两次 GNSS 创新历史

在第一版实现中，建议：

- 使用小型 MLP
- 使用固定维数统计特征
- 尽量避免高维长序列输入，以降低数据需求与过拟合风险

## 9. 监督标签定义

假设训练阶段存在参考真值轨迹。

设参考轨迹在 GNSS 量测空间中对应的理想量测为

\[
\mathbf{z}_{k,\text{ref}}^{g}
\]

则可将真实 GNSS 扰动定义为

\[
\mathbf{d}_{k}^{b,*} = \mathbf{z}_{k}^{g} - \mathbf{z}_{k,\text{ref}}^{g}
\]

等价地，也可在创新域中定义为

\[
\mathbf{d}_{k}^{b,*} = \mathbf{r}_k - \mathbf{r}_{k,\text{ideal}}
\]

其中

\[
\mathbf{r}_{k,\text{ideal}} = \mathbf{z}_{k,\text{ref}}^{g} - h(\hat{\mathbf{x}}_k)
\]

因此，训练目标不是直接回归位置本身，而是回归 GNSS 更新中的未建模偏差项。

## 10. 损失函数设计

基础损失项为扰动状态回归误差：

\[
\mathcal{L}_{\text{dist}} = \left\| \hat{\mathbf{d}}_k^{b} - \mathbf{d}_{k}^{b,*} \right\|_2^2
\]

为避免网络输出过大补偿项，引入正则项：

\[
\mathcal{L}_{\text{reg}} = \left\| \hat{\mathbf{d}}_k^{b} \right\|_2^2
\]

则总损失可写为

\[
\mathcal{L} =
\mathcal{L}_{\text{dist}} + \lambda \mathcal{L}_{\text{reg}}
\]

进一步地，还可加入创新一致性损失：

\[
\mathcal{L}_{\text{upd}} = \left\| \tilde{\mathbf{r}}_k - \mathbf{r}_{k,\text{ideal}} \right\|_2^2
\]

此时总损失为

\[
\mathcal{L} = \mathcal{L}_{\text{dist}} + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{upd}}
\]

## 11. 在线推理流程

在每个 GNSS 更新时刻 \(k\)，算法流程如下：

1. 按 KF-GINS 原始方法完成状态传播与协方差传播
2. 计算当前 GNSS 创新 \(\mathbf{r}_k\)
3. 构造实时特征向量 \(\mathbf{u}_k\)
4. 通过网络推理得到扰动状态估计

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

5. 对创新进行校正

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

6. 使用校正后的创新执行标准 ESKF 更新

由此可见，本文方法只在 GNSS 量测更新前增加一个轻量补偿模块，不破坏原有滤波结构。

## 12. 扰动状态的物理解释

本文提出的 \(\mathbf{d}_k^{b}\) 可理解为：

- 城市退化环境下 GNSS 量测误差的低维潜在表示
- 与位置量测通道直接对应的、具有物理意义的补偿项
- 随场景变化和时间变化的未建模扰动

因此，不应将其表述为：

- 全部 GNSS 误差的完整建模
- 可替代严谨原始观测建模的方法
- 通用环境下的统一固定偏差

## 13. 与现有常见方法的区别

### 13.1 与自适应协方差方法的区别

传统自适应协方差方法的核心形式为

\[
\mathbf{R}_k \leftarrow g_{\theta}(\cdot)
\]

而本文方法采用

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^{b}
\]

二者本质区别在于：

- 前者只改变对量测的信任程度
- 后者直接补偿量测中的结构化偏差项

### 13.2 与 GNSS 中断伪观测方法的区别

GNSS 中断辅助方法通常在 GNSS 不可用时构造替代观测。

本文方法则假设：

- GNSS 仍可输出位置观测
- 但其观测存在显著偏差

因此，本文关注的是 `degradation` 而不是 `outage`。

### 13.3 与端到端神经导航方法的区别

端到端方法通常可写为

\[
\hat{\mathbf{x}}_k = f_{\theta}(\text{raw sensors})
\]

而本文方法始终保留：

- 惯导机械编排
- 误差状态传播
- 卡尔曼更新主干

学习模块仅负责估计难以显式建模的扰动项，属于典型的物理先验主导、数据驱动补充框架。

## 14. 论文中可直接使用的核心公式组

本文核心模型可压缩为如下公式：

\[
\mathbf{z}_k^{g} = h(\mathbf{x}_k) + \mathbf{d}_k^{b} + \mathbf{n}_k
\]

\[
\hat{\mathbf{d}}_k^{b} = f_{\theta}(\mathbf{u}_k)
\]

\[
\tilde{\mathbf{r}}_k = \mathbf{z}_k^{g} - h(\hat{\mathbf{x}}_k) - \hat{\mathbf{d}}_k^{b}
\]

\[
\delta \hat{\mathbf{x}}_k = \delta \hat{\mathbf{x}}_k^{-} + \mathbf{K}_k \left( \tilde{\mathbf{r}}_k - \mathbf{H}_k \delta \hat{\mathbf{x}}_k^{-} \right)
\]

这组公式可以作为方法章节的核心表达。
