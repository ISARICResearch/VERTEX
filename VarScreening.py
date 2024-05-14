import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import IsaricDraw as idw
import pandas as pd
import plotly.graph_objs as go
import pycountry
from dash.dependencies import Input, Output, State
import dash
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]

df=pd.read_csv('assets/data/model_death.csv')
results = []
X = sm.add_constant(df.drop('outcome', axis=1))  # Adding a constant for the intercept

for column in X.columns[1:]:  # Skipping the constant term
    model = sm.Logit(df['outcome'], X[[column, 'const']]).fit(disp=0)  # Fit the model
    
    # Calculate Odds Ratio
    odds_ratio = np.exp(model.params[column])
    
    # Calculate Confidence Interval for Odds Ratio
    conf = model.conf_int().loc[column]
    conf_lower, conf_upper = np.exp(conf[0]), np.exp(conf[1])  # Exponentiate CI
    
    # Get p-value
    p_value = model.pvalues[column]
    
    # Append results
    results.append({
        'Variable': column,
        'Odds Ratio': odds_ratio,
        'P-Value': p_value,
        '95% CI Lower': conf_lower,
        '95% CI Upper': conf_upper
    })



# Convert results to a DataFrame
results_df = pd.DataFrame(results)


def filters_controls():
    row = dbc.Row([
        dbc.Col([html.H6("Gender:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='gender-checkboxes_tr',
                    options=[
                        {'label': 'Male', 'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Unknown', 'value': 'U'}
                    ],
                    value=['Male', 'Female', 'U']
                )
            ])
        ], width=2),
        
        dbc.Col([html.H6("Age:", style={'margin-right': '10px'}),
            html.Div([
                
                html.Div([
                    dcc.RangeSlider(
                        id='age-slider_tr',
                        min=0,
                        max=90,
                        step=10,
                        marks={i: str(i) for i in range(0, 91, 10)},
                        value=[0, 90]
                    )
                ], style={'width': '100%'})  # Apply style to this div
            ])
        ], width=3),
        '''
        dbc.Col([html.H6("Country:", style={'margin-right': '10px'}),
                html.Div([

                    html.Div(id="country-display_tr", children="Country:", style={"cursor": "pointer"}),
                    dbc.Fade(
                        html.Div([
                            dcc.Checklist(
                                id='country-selectall_tr',
                                options=[{'label': 'Select all', 'value': 'all'}],
                                value=['all']
                            ),
                            dcc.Checklist(
                                id='country-checkboxes_tr',
                                options=country_dropdown_options,
                                value=[option['value'] for option in country_dropdown_options],
                                style={'overflowY': 'auto', 'maxHeight': '100px'}
                            )
                        ]),
                        id="country-fade_tr",
                        is_in=False,
                        appear=False,
                    )
                ]),
            
        ], width=5),

        dbc.Col([html.H6("Outcome:", style={'margin-right': '10px'}),
            html.Div([
                
                dcc.Checklist(
                    id='outcome-checkboxes_tr',
                    options=[
                        {'label': 'Death', 'value': 'Death'},
                        {'label': 'Censored', 'value': 'Censored'},
                        {'label': 'Discharge', 'value': 'Discharge'}
                    ],
                    value=['Death', 'Censored', 'Discharge']
                )
            ])
        ], width=2)'''
    ])
    return row


def create_screening_modal():
    linegraph_instructions = html.Div([
        html.Div("1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster)"),
        html.Br(),
        html.Div("2. Change datasets using the dropdown (country selections are remembered)"),
        html.Br(),
        html.Div("3. Hover mouse on chart for tooltip data "),
        html.Br(),
        html.Div("4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it."),
        html.Br(),
        html.Div("5. Toggle selected countries on/off by clicking on the legend (far right)"),
        html.Br(),
        html.Div("6. Download button will export all countries and available years for the selected dataset"),    
    ])   


    forest_fig=idw.forest_plot(results_df)
    modal = [
        dbc.ModalHeader(html.H3("Screening of Variables", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

        dbc.ModalBody([
            dbc.Accordion([
                #dbc.AccordionItem(
                #    title="Filters and Controls",  
                #    children=[filters_controls()]
                #),                
                dbc.AccordionItem(
                    title="Insights",  
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([dbc.Col(forest_fig,id='forest_plot_col')]), label='Bivariate analysis'),
                            #dbc.Tab(dbc.Row([dbc.Col(upset_plot,id='upsetTreat_chart_col')]), label='Combination of treatments'),
                            ])
                    ]
                )
            ])
        ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

        dbc.ModalFooter([

            
            html.Div([
                dbc.Button("About", id='modal_patChar_about_popover', color='info', size='sm',style={'margin-right': '5px'}),
                dbc.Button("Instructions", id='modal_patChar_instruc_popover',  size='sm',style={'margin-right': '5px'}),
                dbc.Button("Download", id='modal_patChar_download_popover', size='sm',style={'margin-right': '5px'}),
                #dbc.Button("Close", id="modal_patChar_close_popover",  size='sm')
            ], className='ml-auto'),
                

                                
            dbc.Popover(
                [
                    dbc.PopoverHeader("Instructions", style={'fontWeight':'bold'}),
                    dbc.PopoverBody(linegraph_instructions)
                ],
                #id="modal-line-instructions-popover",
                #is_open=False,
                target="modal_patChar_instruc_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,
                #style={"zIndex":1}
            ),
            
            dbc.Popover(
                [
                    dbc.PopoverHeader("About", style={'fontWeight':'bold'}),
                    dbc.PopoverBody(linegraph_instructions),
                ],
                #id="modal-line-guide-popover",
                #is_open=False,
                target="modal_patChar_about_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,                        
                #style={"zIndex":1}
            ),
            
            dbc.Popover(
                    [
                    dbc.PopoverHeader("Download", style={'fontWeight':'bold'}),
                    dbc.PopoverBody([          
                        html.Div('Raw data'),
                        dbc.Button(".xlsx", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-xls", style={}, size='sm'),
                        dbc.Button(".csv", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-csv", style={}, size='sm'),
                        dbc.Button(".json", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-json", style={}, size='sm'),
                        html.Div('Chart', style={'marginTop':5}),
                        dbc.Button(".pdf", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-pdf", style={}, size='sm'),
                        dbc.Button(".jpg", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-jpg", style={}, size='sm'),
                        dbc.Button(".png", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-png", style={}, size='sm'),
                        dbc.Button(".svg", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-svg", style={}, size='sm'),
                        html.Div('Advanced', style={'marginTop':5,'display':'none'}),
                        dbc.Button("Downloads Area", outline=True, color="secondary", className="mr-1", id="btn-popover-line-download-land", style={'display':'none'}, size='sm'),
                        ]),
                    ],
                    id="download-popover-patChar",                                        
                    target="modal_patChar_download_popover",
                    #style={'maxHeight': '300px', 'overflowY': 'auto'},
                    trigger="legacy",
                    placement="top",
                    hide_arrow=False,
                    
            ), 
            
        ]),
    ]
    return modal    

