import os
import random
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import warnings
from sklearn.tree import ExtraTreeClassifier

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_fscore_support, roc_auc_score


def predict(X1, X2):
    y1 = clf_1.predict(X1)
    y2 = clf_2.predict(X2)

    # proba_supported = supports_proba(clf_1, X1[0]) and supports_proba(clf_2, X2[0])

    # fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
    y_pred = np.asarray([-1] * X1.shape[0])

    for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
        if y1_i == y2_i:
            y_pred[i] = y1_i
        elif y1_i != y2_i:
            y1_probs = clf_1.predict_proba([X1[i]])[0]
            y2_probs = clf_2.predict_proba([X2[i]])[0]
            sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
            # 采用软投票的方法进行预测选择类别
            max_sum_prob = max(sum_y_probs)
            # 选取概率总和最大的索引即预测标签值
            y_pred[i] = sum_y_probs.index(max_sum_prob)

        else:
            # the classifiers disagree and don't support probability, so we guess
            y_pred[i] = random.randint(0, 1)

    # check that we did everything right
    assert not (-1 in y_pred)

    return y_pred


if __name__ == '__main__':
    np.random.seed(200)

    # 导入原始数据
    raw_data = pd.read_csv('C:/Users/Dell/Desktop/'
                           'self-training and co-training/datasets'
                           '/thyroid.csv')

    # 对数据进行数字化预处理
    encoder_x = LabelEncoder()
    raw_data = raw_data.apply(encoder_x.fit_transform)
    raw_data.sample(frac=1).reset_index(drop=True)

    # raw_data.drop(['Time', 'Amount'], axis=1, inplace=True)
    # MMscaller = MinMaxScaler()
    # raw_data = MMscaller.fit_transform(raw_data)
    print(raw_data.iloc[:, -1].value_counts())
    # raw_data.replace({"class": {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}}, inplace=True)
    # print(raw_data.iloc[:, -1].value_counts())

    X_train, X_test = train_test_split(raw_data, train_size=0.7, random_state=42)

    data_feature_DF = X_train.iloc[:, :-1]
    data_class_DF = X_train.iloc[:, -1]

    test_example = X_test.iloc[:, :-1]
    test_label = X_test.iloc[:, -1]

    # 删除空缺值
    X_train.dropna(how='all', inplace=True)
    X_train.sample(frac=1).reset_index(drop=True)
    data_num = len(X_train)

    p_data = X_train[X_train.iloc[:, -1] == 1]
    n_data = X_train[X_train.iloc[:, -1] == 0]
    pn = p_data.shape[0]
    nn = n_data.shape[0]

    # 训练集中的正负样本个数
    print("训练集中隐藏标记前\n正样本数量为：{}\n负样本数量为：{}".format(pn, nn))

    # 按比例分标记样本和未标记样本
    Label_percent = 0.8
    Label_num = int(data_num * Label_percent)
    unlabeled_num = data_num - Label_num
    print("*************************************************************")
    print("隐藏异常样本后训练集中：\n标记样本为：{}\n无标记样本为：{}".format(Label_num, unlabeled_num))

    # 打乱数据
    p_data = p_data.sample(frac=1).reset_index(drop=True)
    n_data = n_data.sample(frac=1).reset_index(drop=True)

    # U：未标记样本  L：标记样本
    # 创建L标记样本集，将30%的正样本和30%的负样本共同构成带标签样本L
    L = p_data[:int(Label_percent * pn)]
    L = L.append(n_data[:int(Label_percent * nn)])
    L = L.sample(frac=1).reset_index(drop=True)
    L_labels = L.iloc[:, -1]

    # 剩下构成无标记样本集
    p_data = p_data[int(Label_percent * pn):]
    p_data = p_data.reset_index(drop=True)
    n_data = n_data[int(Label_percent * nn):]
    n_data = n_data.reset_index(drop=True)

    # p_data+n_data构成U
    U_raw = p_data
    U_raw = U_raw.append(n_data)
    U_raw = U_raw.sample(frac=1).reset_index(drop=True)
    U = U_raw

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

    # 特征选取
    # forest = ExtraTreeClassifier()
    forest = ExtraTreeClassifier(criterion='gini', max_features=2)
    forest.fit(data_feature_DF, data_class_DF)
    # 采用排列重要性分析
    feature_result = permutation_importance(forest, data_feature_DF, data_class_DF,n_repeats=10,random_state= 10)
    feature_rank2 = feature_result.importances_mean
    # feature_rank = forest.feature_importances_
    # 根据特征重要程度进行排序
    # rank_index = feature_rank.argsort()
    rank_index = feature_rank2.argsort()

    # 将样本按照特征分类为两个视图,索引双数为feature1，索引单数为feature2
    fearture_1_index = rank_index[0::2]
    fearture_2_index = rank_index[1::2]

    # 创建两个训练集X1，X2,以及对应标签y
    X1 = L.iloc[:, fearture_1_index]
    X2 = L.iloc[:, fearture_2_index]
    y = L.iloc[:, -1]

    test_X1 = X_test.iloc[:, fearture_1_index]
    test_X2 = X_test.iloc[:, fearture_2_index]
    test_X1 = np.array(test_X1)
    test_X2 = np.array(test_X2)

    F1_score = []
    Precision = []
    Recall = []
    AUC_score = []

    # 迭代循环次数为it
    it = 1
    while len(U) > 0:
        # for i in range(it):
        clf_1 = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, n_jobs=1)
        clf_1.fit(X1, y)
        clf_2 = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, n_jobs=1)
        clf_2.fit(X2, y)

        # 采用clf1对feature_1进行训练预测正负样本概率分布
        prob1 = clf_1.predict_proba(U1.iloc[:, fearture_1_index])
        sort_index_pos1 = []
        sort_index_neg1 = []
        # 对利用clf1预测到的概率样本进行降序排序
        rank_index_1 = np.argsort(-prob1, axis=0)

        # 选取前p个正样本添加到sort_index_pos1中，并设置其标签为1
        for j in range(p):
            sort_index_pos1.append(rank_index_1[j][1])
        temp = U1.loc[sort_index_pos1]
        temp.iloc[:, -1] = temp.iloc[:, -1].replace(0, 1)
        # 将clf1标记的正样本添加到L中
        L = L.append(temp)
        L = L.reset_index(drop=True)

        count = 0
        for j in range(n):
            # 如果选取的索引也在临时正样本中，就舍弃重新选取，保证选取的是可靠负样本
            if rank_index_1[count][0] in sort_index_pos1:
                j -= 1
                count += 1
            else:
                sort_index_neg1.append(rank_index_1[count][0])
                count += 1
        temp = U1.loc[sort_index_neg1]
        temp.iloc[:, -1] = temp.iloc[:, -1].replace(1, 0)
        # 将clf1标记的负样本添加到L中
        L = L.append(temp)
        L = L.reset_index(drop=True)

        # 从U1 中删除自标记的正负样本
        U1.drop(sort_index_pos1, inplace=True)
        U1.drop(sort_index_neg1, inplace=True)
        U1 = U1.reset_index(drop=True)

        # 采用clf2对feature_2的正负样本预测概率分布
        prob2 = clf_2.predict_proba(U1.iloc[:, fearture_2_index])
        sort_index_pos2 = []
        sort_index_neg2 = []
        # clf_2的预测概率分布
        rank_index_2 = np.argsort(-prob2, axis=0)

        for j in range(p):
            sort_index_pos2.append(rank_index_2[j][1])
        # 构建临时正样本temp
        temp = U1.loc[sort_index_pos2]
        temp.iloc[:, -1] = temp.iloc[:, -1].replace(0, 1)
        # 将clf1标记的正样本添加到L中
        L = L.append(temp)
        L = L.reset_index(drop=True)

        count = 0
        for j in range(n):
            if rank_index_2[count][0] in sort_index_pos2:
                j -= 1
                count += 1
            else:
                sort_index_neg2.append(rank_index_2[count][0])
                count += 1
        # 构建临时负样本
        temp = U1.loc[sort_index_neg2]
        temp.iloc[:, -1] = temp.iloc[:, -1].replace(1, 0)
        # 将可靠负样本添加到L中
        L = L.append(temp)
        L = L.reset_index(drop=True)

        U1.drop(sort_index_pos2, inplace=True)
        U1.drop(sort_index_neg2, inplace=True)

        # 从U样本中再选取2*p+2*n个样本添加到缓冲区U1中、
        U1 = U1.append(U[:2 * p + 2 * n])
        U1 = U1.reset_index(drop=True)
        U = U.loc[2 * p + 2 * n:]
        U = U.reset_index(drop=True)

        if len(U) <= 0:
            break

        # 更新X1,X2,y
        X1 = L.iloc[:, fearture_1_index]
        X2 = L.iloc[:, fearture_2_index]
        y = L.iloc[:, -1]

        # # 更新训练分类器
        # clf_1.fit(X1, y)
        # clf_2.fit(X2, y)

        test_pre = predict(test_X1, test_X2)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            test_label, test_pre)
        auc_score = roc_auc_score(test_label, test_pre)

        Precision.append(precision[1])
        Recall.append(recall[1])
        F1_score.append(f1_score[1])
        AUC_score.append(auc_score)

        print("iteration:{}".format(it))
        print("Precision: {}".format(precision[1]))
        print("Recall: {}".format(recall[1]))
        print("F1 score:{}".format(f1_score[1]))
        print("AUC score:{}".format(auc_score))

        print("#####################################################")

        it += 1

    test_pre = predict(test_X1, test_X2)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_label, test_pre)
    auc_score = roc_auc_score(test_label, test_pre)

    print("iteration:{}".format(it))
    print("Precision: {}".format(precision[1]))
    print("Recall: {}".format(recall[1]))
    print("F1 score:{}".format(f1_score[1]))
    print("AUC score:{}".format(auc_score))

    result_data = pd.DataFrame()
    result_data['F1 score'] = F1_score
    result_data['Precision'] = Precision
    result_data['Recall'] = Recall
    result_data['AUC score'] = AUC_score
    outputpath = 'C:/Users/Dell/Desktop/self-training and co-training/experiment results/result.csv'
    result_data.to_csv(outputpath, sep=',', index=False, header=True, float_format='%.4f')
