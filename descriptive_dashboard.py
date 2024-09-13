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
import TreatmentsNew as treat
#import VarScreening as varScr
import redcap_config as rc_config
import genericPanel as genericFuns
import ClinicalPresentation_Demographics as demogra
import ClinicalPresentation_Comorbidities as comorbid
import ClinicalPresentation_Signs as signs
import AdministeredTreatments_Treatments as treatments

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

#patChars.register_callbacks(app)

##################################

path_data='C:/Users/egarcia/OneDrive - Nexus365/Projects/ISARIC3.0/Data Analysis/Descriptive Dashboard/assets/data/'


site_mapping=rc_config.site_mapping
redcap_api_key=rc_config.redcap_api_key
redcap_url=rc_config.redcap_url

buttons=[
    ['Examples','Generic Panel','generic'],
    ["Clinical Presentation","Demographics","demogra"],
    ["Clinical Presentation","Comorbidities","comorbid"],
    ["Clinical Presentation","Signs and Symptoms","signs"],
    ["Administered Treatments","Treatments","treatments"],
    #["Clinical Presentation","Signs and Symptoms","patientChar"],
    #["Characterization","Symptoms and Comorbidities","symptoms"],
    #["Treatments","Treatments","symptoms"],
    #["Risk/Prognosis" ,"Clinical features by patient outcome","feature-outcome"],
    #["Risk/Prognosis" ,"Risk factors for patient outcomes","treat"]
    ]

##################################
#################################



sections=getRC.getDataSections(redcap_api_key)

vari_list=getRC.getVariableList(redcap_api_key,['dates','demog','comor','daily','outco','labs','vital','adsym','inter','treat'])

#df_map=getRC.get_REDCAP_Single_DB(redcap_url, apis_dengue,site_mapping,vari_list)
df_map=getRC.get_REDCAP_Single_DB(redcap_url, redcap_api_key,site_mapping,vari_list)


df_map_count=df_map[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()

unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country').reset_index(drop=True)

country_dropdown_options=[]
for uniq_county in range(len(unique_countries)):
    name_country=unique_countries['slider_country'].iloc[uniq_county]
    code_country=unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append({'label': name_country, 'value': code_country})


#country_dropdown_options = [{'label': 'Colombia', 'value': 'COL'}, {'label': 'United Kingdom', 'value': 'GBR'}]


max_value = df_map_count['usubjid'].max()
# Define the color scale
custom_scale = []
cutoffs = np.percentile(df_map_count['usubjid'], [10,20,30, 40,50, 60,70, 80,90,99, 100]) 
num_colors = len(cutoffs)
colors = idw.interpolate_colors(['0000FF', '00EA66','A7FA00','FFBE00','FF7400','FF3500'], num_colors)

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

    idw.define_menu(buttons,country_dropdown_options)
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
          return  not is_open, patChars.create_modal()
      elif button_id =='{"index":"demogra","type":"open-modal"}':
          return  not is_open, demogra.create_modal()
      elif button_id =='{"index":"comorbid","type":"open-modal"}':
          return not is_open,comorbid.create_modal()
      elif button_id =='{"index":"treat","type":"open-modal"}':
          return not is_open,treat.create_modal()
      elif button_id =='{"index":"generic","type":"open-modal"}':
          return not is_open,genericFuns.create_modal()
      elif button_id =='{"index":"signs","type":"open-modal"}':
          return not is_open,signs.create_modal()
      elif button_id == '{"index":"treatments","type":"open-modal"}':
          return not is_open,treatments.create_modal()
      return not is_open
  
  return is_open,[]



        



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
    #df_map['age']=df_map['age'].astype(int)
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
treat.register_callbacks(app,'treat')
genericFuns.register_callbacks(app,'generic')
demogra.register_callbacks(app,'demogra')
comorbid.register_callbacks(app,'comorbid')
signs.register_callbacks(app,'signs')

if __name__ == '__main__':
    #app.run_server(debug=True, host='0.0.0.0', port='8080')
    app.run_server(debug=True)