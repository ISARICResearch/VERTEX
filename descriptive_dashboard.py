import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import callback_context
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import getREDCapData as getRC
import IsaricDraw as idw
import IsaricAnalytics as ia
import PatientCharacteristics as patChars
import risk_feature_outcome as risk_fo
import risk_factors as risk_factors
#import SymptomsComorbidities as symComor
#import Treatments as treat
#import VarScreening as varScr

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

#patChars.register_callbacks(app)

##################################

path_data='C:/Users/egarcia/OneDrive - Nexus365/Projects/ISARIC3.0/Data Analysis/Descriptive Dashboard/assets/data/'

##################################
#################################


#df_map=pd.read_csv('assets/data/map.csv')

df_map=getRC.read_data_from_REDCAP()

df_map=df_map.dropna()

'''
df_map1=df_map.copy().loc[df_map['country_iso'].isin(['COL'])]
df_map1=df_map1.sample(100)
df_map2=df_map.copy().loc[df_map['country_iso'].isin(['GBR'])]
df_map2=df_map2.sample(100)
df_map=pd.concat([df_map1,df_map2])'''

#print(df_map)
df_map_count=df_map[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()

unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country')
country_dropdown_options = [{'label': row['slider_country'], 'value': row['country_iso']}
                    for index, row in unique_countries.iterrows()]
#country_dropdown_options = [{'label': 'Colombia', 'value': 'COL'}, {'label': 'United Kingdom', 'value': 'GBR'}]


max_value = df_map_count['usubjid'].max()
# Define the color scale
custom_scale = []
cutoffs = np.percentile(df_map_count['usubjid'], [10,20,30, 40,50, 60,70, 80,90,99, 100]) 
num_colors = len(cutoffs)
colors = idw.interpolate_colors(['FF3500','FF7400','FFBE00','A7FA00', '00EA66', '0000FF'], num_colors)

max_value = num_colors  # Assuming the maximum value is 10 for demonstration
custom_scale = []
for i, cutoff in enumerate(cutoffs):
    value_start = 0 if i == 0 else cutoffs[i - 1] / max(df_map_count['usubjid'])
    value_end = cutoff / max(df_map_count['usubjid'])
    color = colors[i]
    custom_scale.append([value_start, color])
    custom_scale.append([value_end, color])
print(df_map_count)
fig = go.Figure(go.Choroplethmapbox(
    geojson="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
    locations=df_map_count['country_iso'],
    z=df_map_count['usubjid'],
    text=df_map_count['slider_country'],
    colorscale=custom_scale,
    zmin=1,
    zmax=df_map_count['usubjid'].max(),
    marker_opacity=0.5,
    marker_line_width=0,
))

mapbox_style = ["open-street-map", "carto-positron"]

fig.update_layout(
    mapbox_style=mapbox_style[1],  # default
    mapbox_zoom=1.7,
    mapbox_center={"lat": 6, "lon": -75},
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
)


app.layout = html.Div([
    dcc.Graph(id='world-map', figure=fig, style={'height': '100vh', 'margin': '0px'}),
    html.Div([
        html.H1("VERTEX - Visual Evidence & Research Tool for EXploration"),
        html.P("Visual Evidence, Vital Answers")
    ], style={'position': 'absolute', 'top': 0, 'left': 10, 'z-index': 1000}),

    idw.define_menu(country_dropdown_options)
])

@app.callback(
  [Output("modal", "is_open"),Output("modal", "children")],
  [Input({'type': 'open-modal', 'index': ALL}, 'n_clicks')],
  [State("modal", "is_open")],
)
def toggle_modal(n,   is_open):
  ctx = callback_context

  if not ctx.triggered:
      print('not trigger')
      button_id = 'No buttons have been clicked yet'
  else:
      button_id = ctx.triggered[0]['prop_id'].split('.')[0]
      print(button_id)
      if button_id =='{"index":"patientChar","type":"open-modal"}':
          return  not is_open, patChars.create_patient_characteristics_modal()
      elif button_id =='{"index":"feature-outcome","type":"open-modal"}':
          return  not is_open, risk_fo.create_modal()
          #return not is_open
      #elif button_id =='{"index":"risk-factor","type":"open-modal"}':
      #    return not is_open,risk_factors.create_modal()
      return not is_open,[]
  
  return is_open,[]


'''
@app.callback(
    Output("modal", "is_open"),
    [Input("open-patient-char", "n_clicks"), 
     Input("open-symptoms", "n_clicks"), 
     #Input("open-treatment", "n_clicks"), 
     #Input("open-screening", "n_clicks")
     ],
    [State("modal", "is_open")],
)
#def toggle_modal(n1, n2, n3, n4,  is_open):
#def toggle_modal(n1, n2, n3,  is_open):
def toggle_modal(n1, n2,   is_open):
    #if n1 or n2 or n3:
    if n1 or n2 :
        return not is_open
    return is_open'''

'''
@app.callback(
    Output("modal", "children"),
    [Input("open-patient-char", "n_clicks"),
     Input("open-symptoms", "n_clicks"),
     #Input("open-treatment", "n_clicks"),
     #Input("open-screening", "n_clicks")
     ],
)
#def update_modal_content(n1, n2, n3, n4):
#def update_modal_content(n1, n2, n3):
def update_modal_content(n1, n2):
    ctx = dash.callback_context
    print("aqui entro 0")
    if not ctx.triggered:
        return [dbc.ModalBody("Please wait")]

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("aqui entro "+button_id)
    
    if button_id == "open-patient-char":
        try:
          return patChars.create_patient_characteristics_modal()
          #return None
        except:

            modal = [


                    dbc.ModalBody([
                     html.H1("Notice: Insufficient patient data to operate analytic pipeline effectively. Additional data collection required", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})
                    ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

                    dbc.ModalFooter([ 
                    ]),
                ]            
            return modal
    elif button_id == "open-screening":
        print('screening')
        #return varScr.create_screening_modal()    
            
        return None
    elif button_id == "open-symptoms":
        print('symComor')
        #return symComor.create_symptoms_comorbidities_modal()
        return []
'''


        



@app.callback(
    Output('world-map', 'figure'),  
    [
        Input('gender-checkboxes', 'value'),
        Input('age-slider', 'value'),
        Input('outcome-checkboxes', 'value'),
        Input('country-checkboxes', 'value')
    ]
)
def update_map(genders, age_range, outcomes,countries):


    #df_map['age']=df_map['age'].astype(int)
    df_map['age']=df_map['age'].astype(float)
    #df_map['age']=np.round(df_map['age'])
    df_map['age']=df_map['age'].astype(int)
    filtered_df = df_map[(df_map['slider_sex'].isin(genders))& 
                     (df_map['age'] >= age_range[0]) & 
                     (df_map['age'] <= age_range[1]) & 
                     (df_map['outcome'].isin(outcomes)) &
                     (df_map['country_iso'].isin(countries)) ]


    if filtered_df.empty:
        fig = go.Figure(go.Choroplethmapbox(
            geojson="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
            
        ))

        mapbox_style = ["open-street-map", "carto-positron"]

        fig.update_layout(
            mapbox_style=mapbox_style[1],  # default
            mapbox_zoom=1.7,
            mapbox_center={"lat": 6, "lon": -75},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        return fig
    df_map_count=filtered_df[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()
    # Define the color scale
    custom_scale = []
    cutoffs = np.percentile(df_map_count['usubjid'], [10,20,30, 40,50, 60,70, 80,90,99, 100]) 
    num_colors = len(cutoffs)
    colors = idw.interpolate_colors(['FF3500','FF7400','FFBE00','A7FA00', '00EA66', '0000FF'], num_colors)

    max_value = num_colors  # Assuming the maximum value is 10 for demonstration
    custom_scale = []
    for i, cutoff in enumerate(cutoffs):
        value_start = 0 if i == 0 else cutoffs[i - 1] / max(df_map_count['usubjid'])
        value_end = cutoff / max(df_map_count['usubjid'])
        color = colors[i]
        custom_scale.append([value_start, color])
        custom_scale.append([value_end, color])

    fig = go.Figure(go.Choroplethmapbox(
        geojson="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
        locations=df_map_count['country_iso'],
        z=df_map_count['usubjid'],
        text=df_map_count['slider_country'],
        colorscale=custom_scale,
        zmin=1,
        zmax=df_map_count['usubjid'].max(),
        marker_opacity=0.5,
        marker_line_width=0,
    ))

    mapbox_style = ["open-street-map", "carto-positron"]

    fig.update_layout(
        mapbox_style=mapbox_style[1],  # default
        mapbox_zoom=1.7,
        mapbox_center={"lat": 6, "lon": -75},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig

@app.callback(
    [Output('country-checkboxes', 'value'),
     Output('country-selectall', 'options'),
     Output('country-selectall', 'value')],
    [Input('country-selectall', 'value'),
     Input('country-checkboxes', 'value')],
    [State('country-checkboxes', 'options')]
)
def update_country_selection(select_all_value, selected_countries, all_countries_options):
    ctx = dash.callback_context

    if not ctx.triggered:
        # Initial load, no input has triggered the callback yet
        return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'country-selectall':
        if 'all' in select_all_value:
            # "Select all" (now "Unselect all") is checked
            return [[option['value'] for option in all_countries_options], [{'label': 'Unselect all', 'value': 'all'}], ['all']]
        else:
            # "Unselect all" is unchecked
            return [[], [{'label': 'Select all', 'value': 'all'}], []]

    elif trigger_id == 'country-checkboxes':
        if len(selected_countries) == len(all_countries_options):
            # All countries are selected manually
            return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]
        else:
            # Some countries are deselected
            return [selected_countries, [{'label': 'Select all', 'value': 'all'}], []]

    return [selected_countries, [{'label': 'Select all', 'value': 'all'}], select_all_value]

@app.callback(
    Output("country-fade", "is_in"),
    [Input("country-display", "n_clicks")],
    [State("country-fade", "is_in")]
)
def toggle_fade(n_clicks, is_in):
    if n_clicks:
        return not is_in
    return is_in
@app.callback(
    Output('country-display', 'children'),
    [Input('country-checkboxes', 'value')],
    [State('country-checkboxes', 'options')]
)

def update_country_display(selected_values, all_options):
    if not selected_values:
        return "Country:"

    # Create a dictionary to map values to labels
    value_label_map = {option['value']: option['label'] for option in all_options}

    # Build the display string
    selected_labels = [value_label_map[val] for val in selected_values if val in value_label_map]
    display_text = ", ".join(selected_labels)

    if len(display_text) > 20:  # Adjust character limit as needed
        return f"Country: {selected_labels[0]}, +{len(selected_labels) - 1} more..."
    else:
        return f"Country: {display_text}"


patChars.register_callbacks(app,'pc')
risk_fo.register_callbacks(app,'risk_features')
risk_factors.register_callbacks(app,'risk_factors')
#symComor.register_callbacks(app)
'''treat.register_callbacks(app)'''
if __name__ == '__main__':
    #app.run_server(debug=True, host='0.0.0.0', port='8080')
    app.run_server(debug=True)