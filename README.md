# data-mining
数据挖掘，推荐系统算法总结
探索了FM和FFM的含义，进行了FM.py部分的修改：
1、在df.interaction_terms 里面，应该是df.reduce_mean()而不是用df.reduce_sum()
2、df.interaction_terms里tf.matmul(self.X, tf.pow(self.v, 2))中的self.X少了一个平方

需要注意：
python3.5，tensorflow版本1.10.1
FM.py设置了解析器，需要使用python FM.py --mode train执行训练过程，使用python FM.py --mode test执行验证过程

安装：
conda install scikit-learn
conda install pandas
conda install tensorflow==1.14

另外：
1、batch_size=128，指的是过多少个样本更新一次模型的参数，相当于每个融合矩阵就是128个样本的结果，128个样本算作一组，得到结果

2、user_id item_id算进去了，不知道为什么，但在FFM中，没有算进去，不应该算进去的！！在FM.py中做修改，加上：cols.remove('user_id')；cols.remove('item_id')

3、FM和FFM的结果都不理想，融合矩阵是都预测为阳性的。
