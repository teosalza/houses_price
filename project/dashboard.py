import gc
from tkinter.ttk import Style
from dash import html, dcc, Dash, dash_table, Input, Output, State
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

app.layout = html.Div(children=[
    html.Div(children=[  
        html.H1(children='Houses Price'),
        html.H4(children="Analisys on houses built from 1872 to 2010"),
       
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
                ],style={"flow-grow":"1"}),
                html.Div(children=[
                 dash_table.DataTable(fn.demo_data.to_dict('records'), [{"name": i, "id": i} for i in fn.demo_data.columns], 
                        style_cell={"color":"black"}),
                ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"2"})
            ],style={"display":"flex","justify-content":"space-between","order":"3"}),
            
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),

        #Statistical info
        html.Div(children=[
            html.H3(children="Stastistics per year houses built"),
            html.Div(children=[
                #container
                html.Div(children=[
                    #House per year building type
                    html.Div(children=[
                        dcc.Graph(
                            id='hpy_graph',
                            figure=fn.hpr                         
                        )
                    ],style={"flex-grow":"1"}),
                    #House per Year average price
                    html.Div(children=[
                        dcc.Graph(
                            id='hpy_ap_graph',
                            figure= fn.hpr_ap
                        )                    
                    ],style={"flex-grow":"2"}),
                ],style={"display":"flex","justify-content":"center"}),
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

                ],style={"padding":"15px","margin-top":"20px"})
            ])
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),
        
        #Statistical info 2
        html.Div(children=[
            html.H3(children="Stastistics price houses built"),
            html.Div(children=[
                #container
                html.Div(children=[
                    #Price per year building price
                    html.Div(children=[
                        dcc.Graph(
                            id='pry_graph',
                            figure=fn.pry                         
                        )
                    ],style={"flex-grow":"1"}),
                    #House per Year average price
                    html.Div(children=[
                        dcc.Graph(
                            id='pry_zn_graph',
                            figure= fn.pry_zn
                        )                    
                    ],style={"flex-grow":"2"}),
                ],style={"display":"flex","justify-content":"center"}),
                #slider
                html.Div(children=[
                     dcc.RangeSlider(id="pry_rg",
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

                ],style={"padding":"15px","margin-top":"20px"})
            ])
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),
        

        #Correlation description
        html.Div(children=[
            html.H3(children="Correlation between features and SalePrice"),
            html.Div(children=[
                #One-hot-encoding correlazion
                html.Div(children=[
                    dcc.Graph(
                            id='ohe_graph',
                            figure=fn.ohe_corr                         
                        )
                ],style={"flex-grow":"1"}),
                #Label encoding correlation
                html.Div(children=[
                    dcc.Graph(
                            id='le_graph',
                            figure=fn.label_enc_corr                         
                        )
                ],style={"flex-grow":"2"}),
                
            ],style={"display":"flex","justify-content":"center"}),
            #Scatter correlation
            html.Div(children=[
                    dcc.Graph(
                            id='scatter_corr_graph',
                            figure=fn.scatter_corr                         
                        )
                ],style={"margin-top":"25px","display":"flex","justify-content":"center"}),
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"20px","background-color":"#252e3f"}),

        #Models application
        html.Div(children=[
            html.H3(children="Application of statistic models"),
            html.Div(children=[
                #Lr model
                html.Div(children=[
                    html.H5(children="Linear Regression Model"),
                    html.Div(children=[
                        dash_table.DataTable(fn.lr_data_errors.to_dict('records'), [{"name": i, "id": i} for i in fn.lr_data_errors.columns], 
                                style_cell={"color":"black"}),
                        ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"4"}),
                    html.H6(children=fn.lr_score,style={"margin-top":"10px"}),
                ]),
                #Rfr model
                html.Div(children=[
                    html.H5(children="Random Forest Regressor Model"),
                    html.Div(children=[
                        dash_table.DataTable(fn.rfr_data_errors.to_dict('records'), [{"name": i, "id": i} for i in fn.rfr_data_errors.columns], 
                                style_cell={"color":"black"}),
                        ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"4"}),
                    html.H6(children=fn.rfr_score,style={"margin-top":"10px"}),
                ]),
                #Dtr model
                html.Div(children=[
                    html.H5(children="Decision Tree Regressor Model"),
                    html.Div(children=[
                        dash_table.DataTable(fn.dtr_data_errors.to_dict('records'), [{"name": i, "id": i} for i in fn.dtr_data_errors.columns], 
                                style_cell={"color":"black"}),
                        ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"4"}),
                    html.H6(children=fn.dtr_score,style={"margin-top":"10px"}),
                ]),
                #Keras model
                html.Div(children=[
                    html.H5(children="Keras Model"),
                    html.Div(children=[
                        dash_table.DataTable(fn.k_data_errors.to_dict('records'), [{"name": i, "id": i} for i in fn.k_data_errors.columns], 
                                style_cell={"color":"black"}),
                        ],style={"padding-right":"50px","padding-left":"50px","flow-grow":"4"}),
                    # html.H6(children=fn.k_score,style={"margin-top":"10px"}),
                ]),
                html.Div(children=[]),
                html.Div(children=[]),
            ],style={"display":"flex","justify-content":"center"}),
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),

        #Estimate your house price
        html.Div(children=[
            html.H3(children="Estimate your house price (with random forest regressor)"),
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        "Year Built: ",dcc.Input(id='yb-input', value='', type='number',min=1873,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Room above grade: ",dcc.Input(id='trag-input', value='', type='number',min=1, max=15,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Fireplaces:",dcc.Input(id='fp-input', value='', type='number',max=3,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Bathrooms:",dcc.Input(id='br-input', value='', type='number',max=4,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "1st Floor sqrft:",dcc.Input(id='fsf-input', value='', type='number',style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Basement sqrft:",dcc.Input(id='bsf-input', value='', type='number',style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Garage area:",dcc.Input(id='ga-input', value='', type='number',style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Garage cars:",dcc.Input(id='gc-input', value='', type='number',max=5,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Above grade living area:",dcc.Input(id='la-input', value='', type='number',style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Overall quality:",dcc.Input(id='oq-input', value='', type='number',min=1,max=10,style={"width":"80px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"}),
                        "Zone: ",dcc.Dropdown(fn.list_mszoning, 'Commercial', id="zn-input",style={"color":"black","width":"350px","margin-right":"20px","border-radius":"8px","margin-bottom":"10px","margin-left":"5px"})
                    ]),
                    html.Div(children=[
                        html.Button('Submit', id='submit-val', n_clicks=0,style={"border":"0px","border-radius":"3px","width":"75px","height":"45px","background-color":"#ababab"}),
                    ],style={"display":"flex","justify-content":"center"}),
                ],style={"width":"85%"}),
                html.H5(children="Test", id="price-result"),
            ],style={"display":"flex","justify-content":"center"}),
        ],style={"padding":"15px", "border":"1px solid rgb(54 54 54)","border-radius":"3px", "margin-bottom":"50px","background-color":"#252e3f"}),
    ],style={"width":"75%"})
], style={"display":"flex", "justify-content":"center"})


@app.callback(Output("hpy_ap_graph",'figure'),[Input("hpr_ap_rg",'value')])
def update_hpy_ap_graph(year_choosen):
    return fn.get_data_hpr_ap(fn.clean_data,year_choosen[0],year_choosen[1])

@app.callback(Output("hpy_graph",'figure'),[Input("hpr_ap_rg",'value')])
def update_hpy_graph(year_choosen):
    return fn.get_data_hpr(fn.clean_data,year_choosen[0],year_choosen[1])

@app.callback(Output("pry_graph",'figure'),[Input("pry_rg",'value')])
def update_pry_graph(year_choosen):
    return fn.get_data_price_per_year(fn.clean_data,year_choosen[0],year_choosen[1])

@app.callback(Output("pry_zn_graph",'figure'),[Input("pry_rg",'value')])
def update_pry_zn_graph(year_choosen):
    return fn.get_data_price_zone(fn.clean_data,year_choosen[0],year_choosen[1])


@app.callback(Output(component_id="price-result",component_property="children"),
                State('yb-input', 'value'),
                State('trag-input', 'value'),
                State('fp-input', 'value'),
                State('br-input', 'value'),
                State('fsf-input', 'value'),
                State('bsf-input', 'value'),
                State('ga-input', 'value'),
                State('gc-input', 'value'),
                State('la-input', 'value'),
                State('oq-input', 'value'),
                State('zn-input', 'value'),
                Input('submit-val', 'n_clicks')
                )
def update_input_selection(yb_val,trag_val,fp_val,br_val,fsf_val,bsf_val,ga_val,gc_val,la_val,oq_val,zn_val,sub):
    data = np.array([fp_val,yb_val,trag_val,br_val,fsf_val,bsf_val,ga_val,gc_val,la_val,oq_val,fn.le_mapping[fn.diz_mszoning[zn_val]]])
    # data = np.array(['2','1939','5','1','1077','991','205','1','1077','5','3'])
    sim_price = fn.predict_random_forest_regressor(data,fn.sc,fn.rfr_model)
    return f'Estimation: {sim_price}'


if __name__ == '__main__':
    app.run_server(debug=True)