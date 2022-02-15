import pandas as pd
import plotly.express as px


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


#global variable for front-end
project_path = 'D:\esperimenti\programming_database\houses_price'
dataset = pd.read_csv(project_path+'\data\\train.csv')
clean_data = clean_dataset(dataset) 
demo_data = clean_data.iloc[0:10,:]

hpr_ap = get_data_hpr_ap(clean_data,1872,2010)
hpr = get_data_hpr(clean_data,1872,2010)