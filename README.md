# 完整代码的使用说明如下：

1. 如果您需要复现该方法首先需要更改原始数据导入路径

   ```py
   raw_data = pd.read_csv('C:/Users/Dell/Desktop/'
                              'self-training and co-training/datasets'
                              '/covertype.csv')
   ```

2. 在分配训练集和测试集时，是以7:3的比例进行分配

   ```py
   X_train, X_test = train_test_split(raw_data, train_size=0.7, random_state=42)
   ```

   

3. 您可以通过如下代码设置固定比例的已标记样本（经过设置的已标记样本数量远远小于原始数据集的标记样本），达到隐藏部分标记样本的效果

   ```py
       Label_percent = 0.2
       
       L = p_data[:int(Label_percent * pn)]
       L = L.append(n_data[:int(Label_percent * nn)])
       L = L.sample(frac=1).reset_index(drop=True)
       L_labels = L.iloc[:, -1]
   
       # 剩下构成无标记样本集
       p_data = p_data[int(Label_percent * pn):]
       p_data = p_data.reset_index(drop=True)
       n_data = n_data[int(Label_percent * nn):]
       n_data = n_data.reset_index(drop=True)
   ```

4. 采用ExtraTreeClassifier进行特征分析，然后采用排列重要性算法对特征重要程度进行排序，将整个数据集分为两个子数据集，用于下面的协同训练

   ```py
   forest = ExtraTreeClassifier(criterion='gini', max_features=2)
       forest.fit(data_feature_DF, data_class_DF)
       # 采用排列重要性分析
       feature_result = permutation_importance(forest, data_feature_DF, data_class_DF,n_repeats=10,random_state= 10)
       feature_rank2 = feature_result.importances_mean
   ```

5. 设置一个大小固定的缓冲区用于存放还未预测的数据

   ```py
   # 设置选取的的正负样本数量分别为p和n
       percent = 0.02
       p = int(pn * percent)
       n = int(nn * percent)
   
       # 从U中选取u个样本创建一个实例缓冲池
       u = 4 * p + 4 * n
       U1 = U[0: u]
       U1 = U1.reset_index(drop=True)
       U = U[u:]
       U = U.sample(frac=1).reset_index(drop=True)
   ```

6. 随后进入循环对缓冲池中的数据集进行预测分析，直至缓冲区中没有数据时退出循环，输出最终结果

## 注意：

1. 对字符型数据集adult，在实验前需要将其转化为数值型
2. 当对ForestCover数据集进行实验时，由于该数据集是含有6种类别，在本文中是将类别5和6归为“1”，类别1-4归为“0”





