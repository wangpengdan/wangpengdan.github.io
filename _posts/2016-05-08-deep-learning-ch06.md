---
layout: post
title: deep learning ch06.deep feedforward networks
date: 2016-05-08 12:00:00
---
> 本博文是Ian Goodfellow, Yoshua Bengio and Aaron Courville写的deep learning的个人读书笔记  
> 原书地址：[deep learning](http://www.deeplearningbook.org/)  
> 第六章地址：[deep feedforward networks](http://www.deeplearningbook.org/contents/mlp.html)

深度前向网络之所以叫这个名字，是由于信息是从输入的$x$到隐藏层，再到输出$y$这样往前的，是没有往后的反馈的。深度前向网络是很多成功的商业项目的基础，同时也是递归神经网络（recurrent neural network，RNN）的基石。深度前向网络受到神经科学的启发，但是目的并不是来模拟大脑。

理解前向网络的一种方法是以线性模型为例，考虑如何克服线性模型的缺陷。  
**线性模型的缺陷在于无法理解多个特征之间的相互作用**，这个缺陷可以通过引入核函数来解决，即$\phi(x)$。  
但是问题是，怎么找到合适的的$\phi(x)$呢？主要有三种方法：

- 使用通用的核函数，比如RBF。对解决高级问题效果不好
- 使用人类挑选的核函数，这个需要大量的时间和经验
- 学习到核函数，深度学习就是这么做的，而且可以兼顾前两个的优点

### **6.1，例子：学习XOR**  
用线性回归模型来学习异或（XOR）是学习不到的。即训练数据特征为$X = \{ [0,0], [0,1], [1,0], [1,1] \}$，label为$Y = \{ 0, 1, 1, 0 \}$，线性模型为$Y = W^TX + b$误差函数为MSE的时候，可以学习到W为0，b为0.5。原因在于前面所讲的**线性模型的缺陷在于无法理解多个特征之间的相互作用**。  
而使用只有一个隐藏层的
