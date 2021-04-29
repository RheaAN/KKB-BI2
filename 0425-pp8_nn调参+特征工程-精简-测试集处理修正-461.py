#!/usr/bin/env python
# coding: utf-8
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/Used_Car")!ls '/content/drive/MyDrive/Colab Notebooks/Used_Car'pip install seaborn
# In[1]:


## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
train_data = pd.read_csv("used_car_train_20200313.csv", sep=' ')
test = pd.read_csv("used_car_testB_20200421.csv", sep=' ')

## 输出数据的大小信息
print('Train data shape:',train_data.shape)
print('TestA data shape:',test.shape)


# In[3]:


## 通过.head() 简要浏览读取数据的形式
train_data.head()


# In[4]:


## 通过 .info() 简要可以看到对应一些数据列名，以及NAN缺失信息
train_data.info()


# In[5]:


## 通过 .columns 查看列名
train_data.columns


# In[6]:


test.info()


# In[7]:


## 通过 .describe() 可以查看数值特征列的一些统计信息
train_data.describe()


# In[8]:


test.describe()


# # 数据清洗

# ## 异常值处理

# In[9]:


train_data['notRepairedDamage'].value_counts()
#train_data['notRepairedDamage'].describe()


# In[10]:


train_data['notRepairedDamage'].replace('-', '0.0', inplace=True)
test['notRepairedDamage'].replace('-', '0.0', inplace=True)


# In[11]:


train_data['power'].describe()

train_data['power'][train_data['power']>600] = 600  # 不要忘记最后['power']
test['power'][test['power']>600] = 600


# In[12]:


train_data['power'].describe()


# In[13]:


test['power'].describe()


# ## 缺失值补全

# In[14]:


print(train_data.isnull().sum())
print(test.isnull().sum())


# In[15]:


train_data['model'].fillna(train_data['model'].mode()[0],inplace = True)
# print(train_data.isnull().sum())
test['model'].fillna(test['model'].mode()[0],inplace = True)
# print(test['model'].isnull().sum())

train_data['bodyType'].value_counts()
train_data['bodyType'].fillna(train_data['bodyType'].mode()[0],inplace = True)
# print(train_data.isnull().sum())
test['bodyType'].fillna(test['bodyType'].mode()[0],inplace = True)

train_data['fuelType'].value_counts()
train_data['fuelType'].fillna(train_data['fuelType'].mode()[0],inplace = True)
# print(train_data.isnull().sum())
test['fuelType'].fillna(test['fuelType'].mode()[0],inplace = True)

train_data['gearbox'].value_counts()
train_data['gearbox'].fillna(train_data['gearbox'].mode()[0],inplace = True)
# print(train_data.isnull().sum())
test['gearbox'].fillna(test['gearbox'].mode()[0],inplace = True)


# In[16]:


# 特征选择
# drop_cols = ['SaleID', 'regDate', 'creatDate', 'offerType', 'price']
drop_cols = ['SaleID', 'name', 'offerType', 'price']
feature_cols = [col for col in train_data.columns if col not in drop_cols]
print(feature_cols)


# In[17]:


# 查看数值类型
#train_data.info()
numerical_cols = train_data[feature_cols].select_dtypes(exclude='object').columns.to_list()
numerical_cols.remove('regDate') # 剔除日期单独处理
numerical_cols.remove('creatDate')
print('numerical_cols :' + str(len(numerical_cols)))
print(numerical_cols)
print('************************************************************')
# 查看分类类型
categorical_cols = train_data[feature_cols].select_dtypes(include='object').columns.to_list()
print('categorical_cols :'+ str(len(categorical_cols)))
print(categorical_cols)
print('************************************************************')
# 查看日期类型
date_cols = ['regDate','creatDate']
print('date_cols :'+ str(len(date_cols)))
print(date_cols)


# In[18]:


# 数值型字段可视化
def plot_continuous(train,feature,target):
    sns.set( palette="muted", color_codes=True)
#     sns.set_style(style = "darkgrid")   #sns.set_style("darkgrid", {"axes.facecolor": "pink"})
#     plt.figure(figsize = (10,8))  # 画布放在这里，就无法绘制多个子图

    # plt.subplot(1, 1, 1)
    g = sns.distplot(train[feature].dropna(), bins=20,color = "b")  # 频数
    g = sns.kdeplot(train[feature].notnull(), ax = g, color = "Blue", shade= False)
    g.set_xlabel(feature) #,fontsize = 15
    g.set_ylabel("Proportion")
    
    # plt.xlim(0,80)
    # plt.ylim(0,0.04)
    plt.grid(True)
    

plt.figure(figsize=(20, 16))
index = 1
for num in numerical_cols:
    plt.subplot(6,4,index)
    plot_continuous(train_data,num,'price')
    index +=1
plt.tight_layout()
plt.show()
# learning ：
# 离散型数值字段：['bodyType','fuelType','gearbox','kilometer'] => 实际上是类别型
# 连续型数值字段：剩下的


# In[19]:


# numerical 剔除 ['model','brand',bodyType','fuelType','gearbox','kilometer']
numerical_cols= [ i for i in numerical_cols if (i not in ['model','brand','bodyType','fuelType','gearbox','kilometer']) ] 
numerical_cols


# In[20]:


# categorical 增加 ['bodyType','fuelType','gearbox','kilometer']
categorical_cols.extend(['model','brand','bodyType','fuelType','gearbox','kilometer'])
categorical_cols


# In[21]:


import warnings
warnings.filterwarnings('ignore')
#X_data.info()
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype('float64')
test['notRepairedDamage'] = test['notRepairedDamage'].astype('float64')


# In[22]:


def plot_discrete(train,feature,target):

    sns.set( palette="muted", color_codes=True)

#     plt.figure(figsize = (20,5)) # 画布放在这里，就无法在for in 内部绘制多个子图

    # plot count for feature
    plt.subplot(7,2,2*index-1)
    w = train[feature].value_counts()
    b = sns.barplot(x=w.index, y=w.values)
    b.set_xlabel(feature,fontsize = 10)
    b.set_ylabel("Count",fontsize = 10)
    b.legend(fontsize = 14,loc = 'upper right')
    del w

    #plot target mean of feature
    plt.subplot(7,2,2*index)
    v = train.groupby(feature,as_index=False)[target].agg({feature + '_'+target+'_mean':'mean'}).reset_index()
    g = sns.barplot(x = feature, y = feature + '_'+target+'_mean',data = v)
    g = g.set_ylabel("Mean")
    del v


plt.figure(figsize=(18, 18))
index = 1
for cat in categorical_cols:
    plot_discrete(train_data,cat,'price')
    index +=1
plt.tight_layout()
plt.show()


# * body Type :车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
# * kilometer 和price 高度负相关 ; notrepairedDamage,gearbox,fuelType 和 price 也有一定关联
# * fuelType :燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
# * model ,brand 不好说

# In[23]:


# 连续型数值字段和目标字段price的相关关系
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = numerical_cols.copy() # 需要copy(),否则numerical_cols会被改掉
num_cols.append('price')
#匿名特征v_0,v_3,v_8,v_12与'price'相关性很高
corr_num = abs(train_data[train_data['price'].notnull()][num_cols].corr())
plt.figure(figsize=(20, 16))
sns.heatmap(corr_num, annot=True)

与price 高度相关，结合上图看分布较正常：
v_3           0.730946
v_12          0.692823
v_8           0.685798
v_0           0.628397
power         0.556400 # 修正异常值后
# In[24]:


# corr_num = abs(train_data[train_data['price'].notnull()][num_cols].corr())
print('Featured hights correlation with Target_column')
print('Feature\t Correlation')
Target_Corr = corr_num["price"] # update target column
# Target_Corr = Target_Corr[np.argsort(abs(Target_Corr), axis = 0)[::-1]][1:] #sort in descending order
Target_Corr = Target_Corr[np.argsort(Target_Corr, axis = 0)[::-1]][1:] #sort in descending order
print(Target_Corr)


# ## 对日期格式进行处理
"""
train_data['regDate'] = pd.to_datetime(train_data['regDate'],format = '%Y%m%d',errors = 'coerce')
print(train_data['regDate'].head(10))

test['regDate'] = pd.to_datetime(test['regDate'],format = '%Y%m%d',errors = 'coerce')
print(test['regDate'].head(10))

train_data['creatDate'] = pd.to_datetime(train_data['creatDate'],format = '%Y%m%d',errors = 'coerce')
print(train_data['creatDate'].head(10))

test['creatDate'] = pd.to_datetime(test['creatDate'],format = '%Y%m%d',errors = 'coerce')
print(test['creatDate'].head(10))

# 时间多尺度

train_data['regDate_year'] = train_data['regDate'].apply(lambda x : int(str(x)[0:4]))
train_data['regDate_month'] = train_data['regDate'].apply(lambda x : int(str(x)[5:7]))
train_data['regDate_day'] = train_data['regDate'].apply(lambda x : int(str(x)[8:10]))
print(train_data[['regDate_year','regDate_month','regDate_day']].head(10))


test['regDate_year'] = test['regDate'].apply(lambda x : int(str(x)[0:4]))
test['regDate_month'] = test['regDate'].apply(lambda x : int(str(x)[5:7]))
test['regDate_day'] = test['regDate'].apply(lambda x : int(str(x)[8:10]))
print(test[['regDate_year','regDate_month','regDate_day']].head(10))

train_data['creatDate_year'] = train_data['creatDate'].apply(lambda x : int(str(x)[0:4]))
train_data['creatDate_month'] = train_data['creatDate'].apply(lambda x : int(str(x)[5:7]))
train_data['creatDate_day'] = train_data['creatDate'].apply(lambda x : int(str(x)[8:10]))
print(train_data[['creatDate_year','creatDate_month','creatDate_day']].head(10))

test['creatDate_year'] = test['creatDate'].apply(lambda x : int(str(x)[0:4]))
test['creatDate_month'] = test['creatDate'].apply(lambda x : int(str(x)[5:7]))
test['creatDate_day'] = test['creatDate'].apply(lambda x : int(str(x)[8:10]))
print(test[['creatDate_year','creatDate_month','creatDate_day']].head(10))

# 时间datediff
train_data['regDate_diff']= (train_data['regDate'] - train_data['regDate'].min()).dt.days
train_data[['regDate','regDate_diff']]

test['regDate_diff']= (test['regDate'] - train_data['regDate'].min()).dt.days  # test['regDate'].min() =  train_data['regDate'].min()
test[['regDate','regDate_diff']]
"""
# In[25]:


"""
feature_cols.append('regDate_year')
feature_cols.append('creatDate_year')
print(feature_cols)

"""
train_data['regDate_year'] = train_data['regDate'].apply(lambda x : int(str(x)[0:4]))
test['regDate_year'] = test['regDate'].apply(lambda x : int(str(x)[0:4]))

train_data['creatDate_year'] = train_data['creatDate'].apply(lambda x : int(str(x)[0:4]))
test['creatDate_year'] = test['creatDate'].apply(lambda x : int(str(x)[0:4]))

# train_data['regDate_year'].value_counts()


# In[26]:


# num_cols.append('car_year')
num_cols.append('regDate_year')
num_cols.append('creatDate_year')
corr_num = abs(train_data[train_data['price'].notnull()][num_cols].corr())
print('Featured hights correlation with Target_column')
print('Feature\t Correlation')
Target_Corr = corr_num["price"] # update target column
Target_Corr = Target_Corr[np.argsort(abs(Target_Corr), axis = 0)[::-1]][1:] #sort in descending order
# Target_Corr = Target_Corr[np.argsort(Target_Corr, axis = 0)[::-1]] #sort in descending order
print(Target_Corr)


# ## 交叉统计特征

# ### categorical_cols-price

# In[27]:


categorical_cols.remove('notRepairedDamage')
categorical_cols.remove('kilometer')
categorical_cols


# In[28]:


#类别特征对价格的统计最大，最小，平均值等等
# selected_categorical_cols = categorical_cols.copy()
selected_categorical_cols = ['brand']
for col in selected_categorical_cols:
    t = train_data.groupby(col,as_index=False)['price'].agg(
        {col+'_count':'count',col+'_price_max':'max',col+'_price_median':'median',
         col+'_price_min':'min',col+'_price_sum':'sum',col+'_price_std':'std',col+'_price_mean':'mean'})
    train_data = pd.merge(train_data,t,on=col,how='left')
    test = pd.merge(test,t,on=col,how='left') # 对test 也用t merge
    del t
# print(train_data.columns)
# print('\n')
# print(test.columns)


# In[29]:


#类别特征对价格的统计最大，最小，平均值等等
# selected_categorical_cols = categorical_cols.copy()
selected_categorical_cols = ['model']
for col in selected_categorical_cols:
    t = train_data.groupby(col,as_index=False)['price'].agg(
        {col+'_count':'count',col+'_price_max':'max',col+'_price_median':'median',
         col+'_price_min':'min',col+'_price_sum':'sum',col+'_price_std':'std',col+'_price_mean':'mean'})
    train_data = pd.merge(train_data,t,on=col,how='left')
    test = pd.merge(test,t,on=col,how='left') # 对test 也用t merge
    del t
# print(train_data.columns)
# print('\n')
# print(test.columns)


# In[30]:


#类别特征对价格的统计最大，最小，平均值等等
# selected_categorical_cols = categorical_cols.copy()
selected_categorical_cols = ['bodyType']
for col in selected_categorical_cols:
    t = train_data.groupby(col,as_index=False)['price'].agg(
        {col+'_count':'count',col+'_price_max':'max',col+'_price_median':'median',
         col+'_price_min':'min',col+'_price_sum':'sum',col+'_price_std':'std',col+'_price_mean':'mean'})
    train_data = pd.merge(train_data,t,on=col,how='left')
    test = pd.merge(test,t,on=col,how='left') # 对test 也用t merge
    del t
# print(train_data.columns)
# print('\n')
# print(test.columns)


# In[31]:


#类别特征对价格的统计最大，最小，平均值等等
# selected_categorical_cols = categorical_cols.copy()
selected_categorical_cols = ['fuelType']
for col in selected_categorical_cols:
    t = train_data.groupby(col,as_index=False)['price'].agg(
        {col+'_count':'count',col+'_price_max':'max',col+'_price_median':'median',
         col+'_price_min':'min',col+'_price_sum':'sum',col+'_price_std':'std',col+'_price_mean':'mean'})
    train_data = pd.merge(train_data,t,on=col,how='left')
    test = pd.merge(test,t,on=col,how='left') # 对test 也用t merge
    del t
# print(train_data.columns)
# print('\n')
# print(test.columns)


# ### kilometer-power

# In[32]:


# kilometer 和price 负相关，power 和price 正相关
kk = ['kilometer','power']
t1 = train_data.groupby(kk[0],as_index=False)[kk[1]].agg(
        {kk[0]+'_'+kk[1]+'_count':'count',kk[0]+'_'+kk[1]+'_max':'max',kk[0]+'_'+kk[1]+'_median':'median',
         kk[0]+'_'+kk[1]+'_min':'min',kk[0]+'_'+kk[1]+'_sum':'sum',kk[0]+'_'+kk[1]+'_std':'std',kk[0]+'_'+kk[1]+'_mean':'mean'})
# t2 = test.groupby(kk[0],as_index=False)[kk[1]].agg(
#         {kk[0]+'_'+kk[1]+'_count':'count',kk[0]+'_'+kk[1]+'_max':'max',kk[0]+'_'+kk[1]+'_median':'median',
#          kk[0]+'_'+kk[1]+'_min':'min',kk[0]+'_'+kk[1]+'_sum':'sum',kk[0]+'_'+kk[1]+'_std':'std',kk[0]+'_'+kk[1]+'_mean':'mean'})

train_data = pd.merge(train_data,t1,on=kk[0],how='left') # train_data 
test = pd.merge(test,t1,on=kk[0],how='left') # test 也用t1 聚合

del t1
# del t2


# ### 'v_3','v_12','v_8','v_0' 组合

# In[33]:


# 效果不好
num_cols = [0,3,8,12]
for i in num_cols:
    for j in num_cols:
        train_data['new'+str(i)+'*'+str(j)]=train_data['v_'+str(i)]*train_data['v_'+str(j)]
        test['new'+str(i)+'*'+str(j)]=test['v_'+str(i)]*test['v_'+str(j)]
for i in num_cols:
    for j in num_cols:
        train_data['new'+str(i)+'+'+str(j)]=train_data['v_'+str(i)]+train_data['v_'+str(j)]
        test['new'+str(i)+'+'+str(j)]=test['v_'+str(i)]+test['v_'+str(j)]

for i in num_cols:
    for j in num_cols:
        train_data['new'+str(i)+'-'+str(j)]=train_data['v_'+str(i)]-train_data['v_'+str(j)]
        test['new'+str(i)+'-'+str(j)]=test['v_'+str(i)]-test['v_'+str(j)]


# ### categorical_cols-与price 高度想关的匿名特征['v_3','v_12','v_8','v_0']
### categorical_cols-与price 高度想关的匿名特征['v_3','v_12','v_8','v_0']
v_cols = ['v_3','v_12','v_8','v_0']
categorical_cols.remove('notRepairedDamage')
categorical_cols.remove('kilometer')
for ccol in categorical_cols:
    for vcol in v_cols:    
        t1 = train_data.groupby(ccol,as_index=False)[vcol].agg(
                {ccol+'_'+vcol+'_count':'count',ccol+'_'+vcol+'_max':'max',ccol+'_'+vcol+'_median':'median',
              ccol+'_'+vcol+'_min':'min',ccol+'_'+vcol+'_sum':'sum',ccol+'_'+vcol+'_std':'std',ccol+'_'+vcol+'_mean':'mean'})
                 
#         t2 = test.groupby(ccol,as_index=False)[vcol].agg(
#             {ccol+'_'+vcol+'_count':'count',ccol+'_'+vcol+'_max':'max',ccol+'_'+vcol+'_median':'median',
#              ccol+'_'+vcol+'_min':'min',ccol+'_'+vcol+'_sum':'sum',ccol+'_'+vcol+'_std':'std',ccol+'_'+vcol+'_mean':'mean'})
             
        train_data = pd.merge(train_data,t1,on=ccol,how='left') # train_data 
        test = pd.merge(test,t1,on=ccol,how='left') # test 也用t1聚合
        del t1
# ### 优化memory
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

train_data = reduce_mem_usage(train_data)
test = reduce_mem_usage(test)
# ## 特征选取

# In[34]:


# 去掉原始的日期自动，添加上新的日期字段
cols = train_data.columns.to_list()
drop_cols = ['SaleID','name', 'regDate', 'creatDate', 'price']
cols = [col for col in train_data.columns if col not in drop_cols]
print(cols)


# In[39]:


# 提取特征列
X_data = train_data[cols]
Y_data = train_data['price']
X_test = test[cols]
X_data.fillna(X_data.median(),inplace = True)
X_test.fillna(X_test.median(),inplace = True)


# In[40]:


X_data.info()


# # 建模

# # 2.神经网络
from sklearn.preprocessing import MinMaxScaler
#特征归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(pd.concat([X_data,X_test]).values)
all_data = min_max_scaler.transform(pd.concat([X_data,X_test]).values)

# 降维
from sklearn import decomposition
cols_len = int(X_data.shape[1]/3)
pca = decomposition.PCA(n_components=cols_len)  # 选取了 1/3 的特征
all_pca = pca.fit_transform(all_data)
X_data = all_pca[:len(X_data)]
X_test = all_pca[len(X_data):]
del all_pca
# Y_data = np.log1p(train_data['price'])  # 取对数看看
# y_pred = np.expm1(clf.predict(X_test)) 

# In[41]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

X_data = mm.fit_transform(X_data) # train_data
X_test = mm.transform(X_test) # test
cols_len = int(X_data.shape[1])


# ## ----------早停法 + 保存模型 - version 2 （适用于jikeyun）

# * 遇到的问题： 之前用训练集得到的MAE 大概在500上下，然后拿来预测测试集，得到的分数在1000多分以上 => 测试集处理有问题

# In[42]:


# https://www.tensorflow.org/tutorials/keras/save_and_load

### 1）设置保存地址
import os
import tensorflow as tf

checkpoint_path = "training_4/cp-{epoch:04d}.ckpt"   # update !!!!!!!!!!!!!!!!!!!!!!!!!
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 2048  # 提取写batch_size

### 2）设置保存的checkpoint
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=10*batch_size) # save uniquely named checkpoints once every five epochs

### 3）设置早停 early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping() # by default, monitor='val_loss' and patience=1


### 4）Train the model with the new callback

from tensorflow import keras
# 搭建模型
model = keras.Sequential([
    # 定义向量的长度：320维度/ 160维度/ 80 维度  ****************************过拟合：减少神经元个数
    keras.layers.Dense(256,activation = 'relu',input_shape = [cols_len]), # 原先：len[cols]；现在是 降维后的
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(64,activation = 'relu'),
    # 最后的输出结果是Price
    keras.layers.Dense(1)
])
# 设置优化器
model.compile(loss = 'mean_absolute_error',optimizer = 'Adam')

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# 模型训练
history = model.fit(X_data,Y_data,
                    batch_size = batch_size,
                    epochs = 80,        #****************************过拟合：调小
                    validation_split=0.2,
                    verbose=2,
#                     callbacks=[early_stopping,cp_callback] # 早停法 + 保存模型
                    callbacks=[cp_callback]   # 保存模型
#                     callbacks=[early_stopping]  # 早停法
                   )  # Pass callback to training


# In[43]:


history = model.fit(X_data,Y_data,
                    batch_size = batch_size,
                    epochs = 80,        #****************************过拟合：调小
                    validation_split=0.2,
                    verbose=2,
#                     callbacks=[early_stopping,cp_callback] # 早停法 + 保存模型
                    callbacks=[cp_callback]   # 保存模型
#                     callbacks=[early_stopping]  # 早停法
                   )  # Pass callback to training


# In[45]:


history = model.fit(X_data,Y_data,
                    batch_size = batch_size,
                    epochs = 80,        #****************************过拟合：调小
                    validation_split=0.2,
                    verbose=2,
#                     callbacks=[early_stopping,cp_callback] # 早停法 + 保存模型
                    callbacks=[cp_callback]   # 保存模型
#                     callbacks=[early_stopping]  # 早停法
                   )  # Pass callback to training


# In[47]:


history = model.fit(X_data,Y_data,
                    batch_size = batch_size,
                    epochs = 80,        #****************************过拟合：调小
                    validation_split=0.2,
                    verbose=2,
#                     callbacks=[early_stopping,cp_callback] # 早停法 + 保存模型
                    callbacks=[cp_callback]   # 保存模型
#                     callbacks=[early_stopping]  # 早停法
                   )  # Pass callback to training


# In[48]:


y_pred = model.predict(X_test)
y_pred

def show_stats(data):
    print('min: ', np.min(data))
    print('max: ', np.max(data))
    # ptp = max - min
    print('ptp: ', np.ptp(data))
    print('mean: ', np.mean(data))
    print('std: ', np.std(data))
    print('var: ', np.var(data))
    
print(show_stats(y_pred))
print(show_stats(Y_data))



"""
score = ？
>>> test data :
min:  -693.50525
max:  98857.414
ptp:  99550.92
mean:  5845.341
std:  7340.823
var:  53887690.0

>>> train data :

min:  11
max:  99999
ptp:  99988
mean:  5923.327333333334
std:  7501.973469876635
var:  56279605.942732885

"""

# 检查模型保存地址
print(os.listdir(checkpoint_dir))

# 获取最新模型地址
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Create a basic model instance
new_model = keras.Sequential([
            # 定义向量的长度：320维度/ 160维度/ 80 维度
            keras.layers.Dense(320,activation = 'relu',input_shape = [len(cols)]),
            keras.layers.Dense(160,activation = 'relu'),
            keras.layers.Dense(80,activation = 'relu'),
            # 最后的输出结果是Price
            keras.layers.Dense(1)
        ])
# 设置优化器
new_model.compile(loss = 'mean_absolute_error',optimizer = 'Adam')


# Evaluate the model
loss  = new_model.evaluate(X_data, Y_data, verbose=2)
print("Untrained model, MAE: {:5.2f}".format(loss))

# Load the previously saved weights
new_model.load_weights(latest)

# Re-evaluate the model
loss = new_model.evaluate(X_data, Y_data, verbose=2)
print("Restored model, accuracy: {:5.2f}".format(loss))

print(new_model.summary())

# 不满意可以继续训练
# In[49]:


#  loss MAE 可视化
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize = (12,12))
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

    
plot_metric(history, 'loss')


# In[50]:


y_pred[y_pred<0] = 11


# # 输出结果

# In[51]:


true_ID = pd.read_csv('./used_car_sample_submit.csv')
true_ID['SaleID']

sub = pd.DataFrame()
sub['SaleID'] = true_ID['SaleID']
sub['price'] = y_pred
sub.to_csv("ans_nn5.csv",index=False)  


# ![image.png](attachment:6e9b8506-4120-4330-9c2a-e7562302861d.png)
