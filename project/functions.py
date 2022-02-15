import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np

'''--------- implementing functions --------'''

#Clean original dataset and choose only more important features
def clean_dataset(dataset):
    nan_features = dataset.isnull().sum().where(dataset.isnull().sum() != 0 ).dropna()
    sale_corr = dataset.corr()['SalePrice'].sort_values()

    #get only features where corr > 0.5 and interesting features for a possible buyer
    top_sale_corr = sale_corr.sort_values().where(sale_corr > 0.45).dropna()
    top_sale_corr_indexes = top_sale_corr.index
    top_sale_corr_indexes = top_sale_corr_indexes.append(pd.Index(["MSZoning"]))

    #Create new dataset and check that NaN values are not present (Drop Column 'GarageYrBlt' and delete rows where 'MasVnrArea' equals to Nan  )
    new_data = dataset[top_sale_corr_indexes]
    new_data = new_data.drop("GarageYrBlt", axis=1)
    new_data = new_data.drop("MasVnrArea",axis=1)
    new_data = new_data.drop("YearRemodAdd",axis=1)
    return new_data

#Graph number of house per year and average price
def get_data_hpr_ap(new_data,start,end):
    houses_per_year = new_data.groupby("YearBuilt").count()["SalePrice"]
    houses_per_year.rename("Total houses built", inplace=True)
    houses_per_year = houses_per_year.where((houses_per_year.index >= start) & (houses_per_year.index <= end)).dropna()

    median_sale_price = new_data[["YearBuilt","SalePrice"]].groupby("YearBuilt").median('SalePrice').sort_values(by="YearBuilt")
    median_sale_price = median_sale_price.iloc[:,0]
    median_sale_price = median_sale_price.where((median_sale_price.index >= start) & (median_sale_price.index <= end)).dropna()

    fig = px.line(
        y=median_sale_price.div(10000).tolist(), 
        x=median_sale_price.index,
        color=px.Constant("Median Price (x 10k)"),
        labels=dict(x="Year", y="Median Price (x 10k)", color="Time Period"),
        title="Numero di case per anno e prezzo medio")
    fig.add_bar(
        y=houses_per_year.values.tolist(), 
        x=median_sale_price.index,
        name="Nr. houses")
    return fig

#Number house per year and tipology of house
def get_data_hpr(new_data,start,end):
    houses_per_year_type = new_data.groupby(["YearBuilt","MSZoning"]).count()
    houses_per_year_type.reset_index(level=1, inplace=True)
    houses_per_year_type.reset_index(level=0, inplace=True)
    houses_per_year_type = houses_per_year_type[["YearBuilt","MSZoning","SalePrice"]].rename(columns={'SalePrice':"count"})
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['A'],'Agriculture')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['C (all)'],'Commercial')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['FV'],'Floating Village Residential')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['I'],'Industrial')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['RH'],'Resid. High Density')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['RL'],'Resid. Low Density')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['RP'],'Resid. Low Density Park')
    houses_per_year_type['MSZoning'] = houses_per_year_type['MSZoning'].replace(['RM'],'Resid. Medium Density')

    filter_hpr = (houses_per_year_type.index >= start) & (houses_per_year_type.index <= end)
    houses_per_year_type =  houses_per_year_type[ (houses_per_year_type["YearBuilt"]>= start) & (houses_per_year_type["YearBuilt"] <= end)]
    fig2 = px.bar(houses_per_year_type, 
        x="YearBuilt",
        y="count", 
        color="MSZoning", 
        title="Numero di case per zona ",
        labels={'count':'Numero Case','YearBuilt':'Anno construzione'})
    return fig2

#Correlazion graph with one-hot-encodig
def get_data_ohe(new_data):
    sale_corr = new_data.corr()['SalePrice'].sort_values()
    ohe_data = pd.get_dummies(new_data)
    ohe_data_corr = ohe_data.corr()['SalePrice'].sort_values().where(sale_corr > 0.45).dropna()
    ohe_enc_data = pd.get_dummies(new_data)
    corr_map = ohe_enc_data.corr().sort_values("SalePrice")["SalePrice"].reset_index(level=0).rename(columns={"index":"Feature"})
    fig3 = px.bar(corr_map, x='Feature', y='SalePrice',
                hover_data=['SalePrice', 'Feature'], color='SalePrice', title="One hot encoded correlation")
    return fig3

#Correlation graph with label encoding
def get_data_label_enc(new_data):
    data_le = _create_label_encoded_data_(new_data)
    corr_map_le = data_le.corr().sort_values("SalePrice")["SalePrice"].reset_index(level=0).rename(columns={"index":"Feature"})
    fig3 = px.bar(corr_map_le, x='Feature', y='SalePrice',
                hover_data=['SalePrice', 'Feature'], color='SalePrice', title="Label encoded correlation")
    return fig3

#Label encoded data
def _create_label_encoded_data_(new_data):
    le = LabelEncoder()
    data_le = new_data.copy()
    for f in new_data:
        if(new_data[f].dtypes == 'object'):
            data_le[f] = le.fit_transform(new_data[f].values)
    return data_le

#Correlazion scatter
def get_data_correlazion_scatter(new_data):
    data_le = _create_label_encoded_data_(new_data)
    fig4 = make_subplots(rows=4, cols=3,subplot_titles=("Fireplaces","YearBuilt","FullBath","TotRmsAbvGrd","1stFlrSF","TotalBsmtSF","GarageArea","GarageCars","GrLivArea","OverallQual","SalePrice","MSZoning" ))
    cont = 0
    for i in range (4):
        for j in range (3):
            fig4.add_trace(
                go.Scatter(x=data_le.iloc[:,cont], y=data_le["SalePrice"],mode="markers"),
                row=i+1, col=j+1
            )
            cont+=1

    fig4.update_layout(height=1200, width=1100, title_text="Correlation scatter matrix")
    return fig4

#Evaluation of models
def _get_errors_performace_(y_test, y_pred, model):
    MAE='{0:.2f}'.format(mean_absolute_error(y_test,y_pred)) 
    MSE='{0:.2f}'.format(mean_squared_error(y_test,y_pred))  
    RMSE='{0:.2f}'.format(np.sqrt(mean_squared_error(y_test,y_pred)))  
    return pd.DataFrame({"Eval Param":["MAE","MSE","RMSE"],"Result":[MAE,MSE,RMSE]})

#Simulation score on test data
def get_scores(model,x_test,y_test):
    return "Total test score: "+ str(model.score(x_test, y_test))

#Get train amd test data
def get_train_test_data(new_data):
    data_le = _create_label_encoded_data_(new_data)
    sc = StandardScaler()
    X_sc = sc.fit_transform(data_le.drop("SalePrice", axis=1))
    x_final = X_sc
    y_final = data_le["SalePrice"]
    return train_test_split(x_final, y_final, test_size=0.2, random_state=2)

#Compute LinearRegression model
def get_linear_regression_data(x_train, y_train, x_test):
    modelLR = LinearRegression()
    modelLR.fit(x_train, y_train)
    y_pred = modelLR.predict(x_test)
    return _get_errors_performace_(y_test,y_pred,"LinearRegression"), y_pred, modelLR

#Computer RanbdomForestRegressor model
def get_random_forest_regressor(x_train, y_train, x_test):
    modelRFR = RandomForestRegressor()
    modelRFR.fit(x_train, y_train)
    y_pred = modelRFR.predict(x_test)
    return _get_errors_performace_(y_test,y_pred,"RandomForestRegressor"), y_pred, modelRFR
    
#Compute DecisionTreeRegressor
def get_decision_tree_regressor(x_train, y_train, x_test):
    modelDTR = DecisionTreeRegressor()
    modelDTR.fit(x_train, y_train)
    y_pred = modelDTR.predict(x_test)
    return _get_errors_performace_(y_test,y_pred,"DecisionTreeRegressor"), y_pred, modelDTR


#global variable for front-end
project_path = 'D:\esperimenti\programming_database\houses_price'
dataset = pd.read_csv(project_path+'\data\\train.csv')
clean_data = clean_dataset(dataset) 
demo_data = clean_data.iloc[0:10,:]
list_mszoning = ['Resid. Low Density', 'Resid. Medium Density', 'Commercial', 'Floating Village Residential', 'Resid. High Density']
diz_mszoning={
    "RL":"Resid. Low Density",
    "RM":"Resid. Medium Density",
    "C (all)":"Commercial",
    "FV":"Floating Village Residential",
    "RH":"Resid. High Density"
}

hpr_ap = get_data_hpr_ap(clean_data,1872,2010)
hpr = get_data_hpr(clean_data,1872,2010)
ohe_corr = get_data_ohe(clean_data)
label_enc_corr = get_data_label_enc(clean_data)
scatter_corr = get_data_correlazion_scatter(clean_data)

x_train, x_test, y_train, y_test = get_train_test_data(clean_data)
#Linear Regression
lr_data_errors, lr_y_pred, lr_model = get_linear_regression_data(x_train,y_train, x_test)
lr_score = get_scores(lr_model, x_test, y_test)
#Random Forest Regressor
rfr_data_errors, rfr_y_pred, rfr_model = get_random_forest_regressor(x_train,y_train, x_test)
rfr_score = get_scores(rfr_model, x_test, y_test)
#Decision Tree Regressor
dtr_data_errors, dtr_y_pred, dtr_model = get_decision_tree_regressor(x_train,y_train, x_test)
dtr_score = get_scores(dtr_model, x_test, y_test)





