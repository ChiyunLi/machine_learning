#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:23:59 2018

@author: rikigei
"""

from util import get_dummies, detect_str_columns, model_testRF, results_summary_to_dataframe, plot_confusion_matrix,logistic_model,logistic_importance,logistic_conf
from util import model, profit_linechart, profit_linechart_all,

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd

# ----設定繪圖-------
import matplotlib.pyplot as plt
import seaborn as sns 


# 讀取電商資料
try:
    data = pd.read_csv('bank-full.csv',encoding = 'cp950')
except:
    data = pd.read_csv('bank-full_utf8.csv')
    
# 看看前5筆資料
data.head()

# 看看前變數型態 (查看欄位資訊)
data.info()


# 偵測有字串的欄位
str_columns = detect_str_columns(data)
print(str_columns)

# 使用獨熱編碼
dataset = get_dummies(str_columns, data)

# 確認全部都是數字
dataset.info()

# 切分資料集
X =dataset.drop(columns=['buy'])
y =dataset['buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# 來看看各自的維度
X_train.shape
y_train.shape

X_test.shape
y_test.shape


# 設定參數
sales_price = 3500
marketing_expense = 300
product_cost = 1800


# logistics regression and confusion matrix
conf_logist = logistic_conf(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        plot_name = 'logistic_regression')


# all_profit 
all_profit = sales_price * conf_logist[1,::].sum() - product_cost * conf_logist[1,::].sum() - marketing_expense * conf_logist.sum()

all_df = pd.DataFrame({
        '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
        '金額' : [sales_price, product_cost, marketing_expense, '-'],
        '目標對象' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
        '小計' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
        })
print('------------------全市場行銷利潤矩陣------------------')
print(all_df)

all_df.to_excel('全市場行銷利潤矩陣.xlsx')  
print('全市場行銷利潤矩陣.xlsx saved')

# logistic regression model_profit 
print(conf_logist)

model_profit = sales_price * conf_logist[1,1] - product_cost * conf_logist[1,1] - marketing_expense * conf_logist[::,1].sum()
model_df = pd.DataFrame({
        '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
        '金額' : [sales_price,product_cost, marketing_expense, '-'],
        '目標對象' : [conf_logist[1,1], conf_logist[1,1], conf_logist[::,1].sum(), '-' ],
        '小計' : [sales_price * conf_logist[1,1], product_cost * conf_logist[1,1],conf_logist[::,1].sum() * marketing_expense,    model_profit],
        })  

print('------------------logistic_regression模型行銷利潤矩陣------------------') 
print(model_df)

model_df.to_excel('logistic_regression模型行銷利潤矩陣.xlsx')
print('logistic_regression模型行銷利潤矩陣.xlsx saved')

# profit comparison
if model_profit - all_profit > 0 :
    print( '------------------模型相對全市場行銷來說【賺錢】------------------' )
    print( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
else:
    print( '------------------模型相對權市場行銷來說【賠錢】------------------' )
    print( '模型比全市場行銷損失 $' + str(model_profit - all_profit) )


# importance_var
importance_var = logistic_importance(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        plot_name = 'logistic_regression'
                   )
importance_var.to_excel('log_impVar.xlsx')

# 實際與預期獲利data frame
# y_test_df=pd.DataFrame(y_test)
y_test_df

# 應注意之變數

result_df

# Summary
all_df, model_profit_df, result_df, y_test_df = logistic_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        plot_name = 'logistic_regression')
    

#-------------RandomForestClassifier-------------

# RF and confusion matrix
model_testRF(RandomForestClassifier(n_estimators = 100, random_state = 0), 
             X_train,y_train,X_test,y_test,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800,
                    plot_name = 'RandomForestClassifier')

# Summary
all_df, model_profit_df, res_data, RF_y_test_df = model(
                                                    clf = RandomForestClassifier(n_estimators = 100, random_state = 0),
                                                    X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,
                                                    sales_price = 3500,
                                                    marketing_expense = 300,
                                                    product_cost = 1800,
                                                    plot_name = 'Random_Forest')

# 預期獲利最佳化模型與閥值折線圖
profit_linechart(y_test_df=RF_y_test_df,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800,
                    plot_name = 'RandomForest')      


# ---------------------XGBClassifier-----------------------

# 基本預測 ＃

# 設定xgb分類模型
clf = XGBClassifier(n_estimators=300 ,random_state = 0, nthread = 8, learning_rate=0.009)
model_xgb = clf.fit(X_train, y_train, verbose=True,eval_set=[(X_train, y_train), (X_test, y_test)])

# 進行預測
y_pred = model_xgb.predict(X_test)
y_pred_prob = model_xgb.predict_proba(X_test)[:,1]

# 製作實際與預測機率表
XGBClassifier_test_df=pd.DataFrame(y_test)
XGBClassifier_test_df['XGBClassifier_pred'] = y_pred_prob
print(XGBClassifier_test_df)


all_df, model_profit_df, res_data, xgboost_y_test_df = model(
         clf = XGBClassifier(n_estimators = 300, random_state = 0),
                                                    X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,
                                                    sales_price = 3500,
                                                    marketing_expense = 300,
                                                    product_cost = 1800,
                                                    plot_name = 'xgboost'
        )

# 預期獲利最佳化模型與閥值折線圖
profit_linechart(
        y_test_df=xgboost_y_test_df,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800,
                    plot_name = 'xgboost'  )



# 三模型獲利比較
profit_linechart_all(y_test_df= [xgboost_y_test_df, RF_y_test_df, y_test_df] ,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800
                    )
