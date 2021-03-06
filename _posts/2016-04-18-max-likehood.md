---
layout: post
title: 最大似然
date: 2016-04-18 12:00:00
---
之前一直知道最大似然的大概意思，但是要讲清楚的话又做不到，所以在这里全面的回顾一下。

### **一，通俗解释**

> 最大似然通俗点来解释就是：利用已知样本，反推最有可能（最大概率）抽取出已知样本的参数值

### **二，数学解释**

$$ P(x_{1}, x_{2}, x_{3}) = f_{D}(x_{1}, x_{2}, x_{3}|\theta) \tag{1} $$

$$ lik(\theta) = \prod_{x \in D_{c}}f_{D}(x_{1}, x_{2}, x_{3}|\theta) \tag{2} $$

假设X的label可以为$x_1, x_2, x_3$，式1为$X$的概率函数，$\theta$的值未知，$D_c$为从$D$中选取的x的集合。

为了预估$\theta$的值，可以用$D_c$数据，即使得取到$D_c$概率最大的 $ \theta $ 即为实际$\theta$的值。如式2所示，式2即为似然函数。预估$\theta$即为使得式2取最大值。

乘法转换成加法比较好求解。所以对似然函数两边取log。

$$ log(lik(\theta)) = \sum_{i=0}^{c}log(f_{D}(x_1,x_2,x_3|\theta)) $$
