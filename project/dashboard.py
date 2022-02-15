from tkinter.ttk import Style
from dash import html, dcc, Dash, dash_table, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,mutual_info_classif,chi2
from matplotlib import pyplot
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import functions as fn







#Plotly dashboard settings
app = Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])
app.title ="Houses Price"
defaultLayout = go.Layout(
      autosize=True)

#call to external file function


df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.Div(children=[  
        html.H1(children='Houses Price'),
        html.H4(children="Analisys on houses development from 1872 to 2010"),
       
        #Description table
        html.Div(children=[
            html.Div(children="Choosen features from dataset"),
            html.Div(children=[
                html.Div(children=[
                    html.H4(children="Features list:"),
                    html.H6(children="  -Fireplaces: Number of fireplaces",style={"padding-left":"15px"}),
                    html.H6(children="  -YearBuilt: Original construction date",style={"padding-left":"15px"}),
                    html.H6(children="  -TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)",style={"padding-left":"15px"}),
                    html.H6(children="  -FullBath: Full bathrooms above grade",style={"padding-left":"15px"}),
                    html.H6(children="  -1stFlrSF: First Floor square feet",style={"padding-left":"15px"}),
                    html.H6(children="  -TotalBsmtSF: Total square feet of basement area",style={"padding-left":"15px"}),
                    html.H6(children="  -GarageArea: Size of garage in square feet",style={"padding-left":"15px"}),
                    html.H6(children="  -GarageCars: Size of garage in car capacity",style={"padding-left":"15px"}),
                    html.H6(children="  -GrLivArea: Above grade (ground) living area square feet",style={"padding-left":"15px"}),
                    html.H6(children="  -OverallQual: Rates the overall material and finish of the house",style={"padding-left":"15px"}),
                    html.H6(children="  -SalePrice: ",style={"padding-left":"15px"}),
                    html.H6(children="  -MSZoning: Identifies the general zoning classification of the sale.",style={"padding-left":"15px"}),
                ],style={"flow-grow":"0"}),
                html.Div(children=[
                 dash_table.DataTable(fn.demo_data.to_dict('records'), [{"name": i, "id": i} for i in fn.demo_data.columns], 
                        style_cell={"color":"black"}),
                ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"4"})
            ],style={"display":"flex","justify-content":"space-between","order":"5"}),
            
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),

        html.Div(children=[
            html.Div(children=""),
            html.Div(children=[
                #container
                html.Div(children=[
                    #House per year building type
                    html.Div(children=[
                        dcc.Graph(
                            id='hpy_graph',
                            figure=fig                         
                        )
                    ],style={"flex-grow":"1"}),
                    #House per Year average price
                    html.Div(children=[
                        dcc.Graph(
                            id='hpy_ap_graph',
                            figure= fn.hpr_ap
                        )                    
                    ],style={"flex-grow":"2"}),
                ],style={"display":"flex","justify-content":"center","padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"20px","background-color":"#252e3f"}),
                #slider
                html.Div(children=[
                     dcc.RangeSlider(id="hpr_ap_rg",
                                    min=1872,
                                    max=2010,
                                    step=5,
                                    marks={
                                        1872: {'label': '1872', 'style': {'color': 'white'}},
                                        1900: {'label': '1900', 'style': {'color': 'white'}},
                                        1925: {'label': '1925', 'style': {'color': 'white'}},
                                        1950: {'label': '1950', 'style': {'color': 'white'}},
                                        1975: {'label': '1975', 'style': {'color': 'white'}},
                                        2000: {'label': '2000', 'style': {'color': 'white'}},
                                        2010: {'label': '2010', 'style': {'color': 'white'}},
                                    },
                                    value=[1872,2010])

                ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"})
            ])
        ]),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),
        dcc.Graph(
            id='example-graph',
            figure=fig
        )],style={"width":"75%"})
  
], style={"display":"flex", "justify-content":"center"})


@app.callback(Output("hpy_ap_graph",'figure'),[Input("hpr_ap_rg",'value')])
def update_hpy_ap_graph(year_choosen):
    return fn.get_data_hpr_ap(fn.clean_data,year_choosen[0],year_choosen[1])

@app.callback(Output("hpy_graph",'figure'),[Input("hpr_ap_rg",'value')])
def update_hpy_graph(year_choosen):
    return fn.get_data_hpr(fn.clean_data,year_choosen[0],year_choosen[1])




if __name__ == '__main__':
    app.run_server(debug=True)