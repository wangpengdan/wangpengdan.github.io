---
layout: post
title: gbdt和gbrank笔记
date: 2016-07-10 12:00:00
---
>gbdt和gbrank是在decision tree基础上建立的用于分类和回归的机器学习算法，gbdt和gbrank属于**boosting算法**，即由一堆弱模型组合而成的强的模型。gbdt和gbrank的区别在于，gbdt是pointwise的，具有**点回归**性质，而gbrank是pairwise的，不具有点回归性质。  
>本文主要参考：  
>1)[A regression framework for learning ranking functions using relative relevance judgments](http://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf)  
>2)[Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

## GBDT
GBDT(gradient boosting decision tree)从名称可知是基于decision tree和梯度建立的boosting机器学习算法。

### 原理

以回归问题为例，定义损失函数为MSE，即为:

$$ \mathop{min}\limits_{h\in H}L(h) = \mathop{min}\limits_{h\in H}\sum_{i=1}^{n}(g_i - h(x_i))^2 $$  

可以使用梯度下降的方法来求解$L(h)$:  

$$ h_{k+1}(x) = h_k(x) - \alpha_{k}\triangledown L(h_k(x)) $$  

对单一的$x_i$求$\triangledown L(h_k(x_i))$为:  

$$ \triangledown L(h_k(x_i)) = -(g_i - h_k(x_i)) $$  

gbdt的学习过程直观来说是每一次拟合最终结果与学习到的模型的残差学习一颗新的子树，最终所有子数加和即为最终模型。由以上式子对gbdt的解释为：$h_k(x)$模型的效果还不够好的情况下，需要进一步优化模型，这里的做法并不是直接去优化$h_k(x)$，而是基于使用$g_i - h_k(x_i)$的**残差**去学习一个新的决策树，以此类推。这样做刚好保证学习过程是沿着梯度下降的方向进行的，保证了可以收敛。  
一个简单的例子为：假如有一堆训练数据，label值为年龄，其中一个训练数据的label值为18岁，训练第一棵树的得到的年龄是16，即用18-16=2训练第二棵树，最终第二棵树得到的年龄为2.5，则用2-2.5=-0.5训练第三棵树，最终第三棵树得到的年龄是-0.3。GBDT到第三棵树训练结束的话，实际年龄是18岁，模型得到的年龄是16+2.5-0.3=18.2岁。  

### 算法
算法的步骤为：  
1)初始化第一棵子树为常数，这里初始化为$h_0(x)=\sum_{i=1}^ny_i/N$，N为训练数据的数量    
2)循环$k\in\lbrace1...M\rbrace$，M为树的棵树   
a)对于每一个训练数据i，计算负梯度$g_i - h_k(x_i)$  
b)使用残差学习一棵回归树$\sum_{j=1}^{J_k} \gamma_{jk}I(x \in R_{jk})$   
c)$h_{k}(x) = h_{k-1}(x) + \eta \sum_{j=1}^{J_k} \gamma_{jk}I(x \in R_{jk})$，$\eta$为学习率。  

## GBRANK
GBRANK与GBDT的不同点为：gbrank是pairwise的，gbdt是pointwise的。所以GBRANK模型训练是需要组pair的。
### 原理
用$
### 算法
