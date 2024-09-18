import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import plotly.graph_objs as go
import sys
import IsaricDraw as idw
import redcap_config as rc_config
import getREDCapData as getRC
from insight_panels import *
from insight_panels.__init__ import __all__ as ip_list

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True)

##################################

redcap_url = rc_config.redcap_url
redcap_api_key = rc_config.redcap_api_key
site_mapping = rc_config.site_mapping

ip_list = [
    module for name, module in sys.modules.items()
    if name.startswith('insight_panels.') & (name.split('.')[-1] in ip_list)]

# ip_list = [generic, demog_comor]

buttons = [
    [ip.research_question, ip.panel_title, ip.suffix]
    for ip in ip_list]

for insight_panel in ip_list:
    insight_panel.register_callbacks(app, insight_panel.suffix)

geojson = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'

##################################
#################################

sections = getRC.getDataSections(redcap_url, redcap_api_key)

vari_list = getRC.getVariableList(
    redcap_url, redcap_api_key,
    ['dates', 'demog', 'comor', 'daily', 'outco', 'labs', 'vital',
        'adsym', 'inter', 'treat'])

df_map = getRC.get_REDCAP_Single_DB(
    redcap_url, redcap_api_key, site_mapping, vari_list)

df_map_count = df_map[['country_iso', 'slider_country', 'usubjid']].groupby(
    ['country_iso', 'slider_country']).count().reset_index()

unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates(
    ).sort_values(by='slider_country').reset_index(drop=True)

country_dropdown_options = []
for uniq_county in range(len(unique_countries)):
    name_country = unique_countries['slider_country'].iloc[uniq_county]
    code_country = unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append({
        'label': name_country,
        'value': code_country})

max_value = df_map_count['usubjid'].max()
# Define the color scale
custom_scale = []
cutoffs = np.percentile(df_map_count['usubjid'], [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100])  # TODO: remove 99?
num_colors = len(cutoffs)
colors = idw.interpolate_colors(
    ['0000FF', '00EA66', 'A7FA00', 'FFBE00', 'FF7400', 'FF3500'], num_colors)

max_value = num_colors  # Assuming the maximum value is 10 for demonstration
custom_scale = []
for i, cutoff in enumerate(cutoffs):
    value_start = (
        0 if i == 0 else (cutoffs[i - 1] / max(df_map_count['usubjid'])))
    value_end = cutoff / max(df_map_count['usubjid'])
    color = colors[i]
    custom_scale.append([value_start, color])
    custom_scale.append([value_end, color])
print(df_map_count)
fig = go.Figure(go.Choroplethmapbox(
    geojson=geojson,
    locations=df_map_count['country_iso'],
    z=df_map_count['usubjid'],
    text=df_map_count['slider_country'],
    colorscale=custom_scale,
    zmin=1,
    zmax=df_map_count['usubjid'].max(),
    marker_opacity=0.5,
    marker_line_width=0,
))

mapbox_style = ['open-street-map', 'carto-positron']

fig.update_layout(
    mapbox_style=mapbox_style[1],  # default
    mapbox_zoom=1.7,
    mapbox_center={'lat': 6, 'lon': -75},
    margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
)


app.layout = html.Div([
    dcc.Graph(
        id='world-map', figure=fig,
        style={'height': '100vh', 'margin': '0px'}),
    html.Div([
        html.H1('VERTEX - Visual Evidence & Research Tool for EXploration'),
        html.P('Visual Evidence, Vital Answers')
        ],
        style={'position': 'absolute', 'top': 0, 'left': 10, 'z-index': 1000}),
    idw.define_menu(buttons, country_dropdown_options)
])


@app.callback(
  [Output('modal', 'is_open'), Output('modal', 'children')],
  [Input({'type': 'open-modal', 'index': ALL}, 'n_clicks')],
  [State('modal', 'is_open')],
)
def toggle_modal(n, is_open):
    ctx = callback_context
    if not ctx.triggered:
        # print('not trigger')
        button_id = 'No buttons have been clicked yet'
        output = is_open, []
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # print(button_id)
        insight_panel = [
            ip for ip in ip_list
            if button_id == '{"index":"' + ip.suffix + '","type":"open-modal"}']
        output = not is_open, insight_panel[0].create_modal()
    return output


@app.callback(
    Output('world-map', 'figure'),
    [
        Input('gender-checkboxes', 'value'),
        Input('age-slider', 'value'),
        Input('outcome-checkboxes', 'value'),
        Input('country-checkboxes', 'value')
    ]
)
def update_map(genders, age_range, outcomes, countries):
    df_map['age'] = df_map['age'].astype(float)
    filtered_df = df_map[(
        (df_map['slider_sex'].isin(genders)) &
        (df_map['age'] >= age_range[0]) &
        (df_map['age'] <= age_range[1]) &
        (df_map['outcome'].isin(outcomes)) &
        (df_map['country_iso'].isin(countries)))]

    if filtered_df.empty:
        fig = go.Figure(go.Choroplethmapbox(geojson=geojson))

        mapbox_style = ['open-street-map', 'carto-positron']

        fig.update_layout(
            mapbox_style=mapbox_style[1],  # default
            mapbox_zoom=1.7,
            mapbox_center={'lat': 6, 'lon': -75},
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        )
        return fig

    df_map_count = (
        filtered_df[['country_iso', 'slider_country', 'usubjid']].groupby(
            ['country_iso', 'slider_country']).count().reset_index())
    # Define the color scale
    custom_scale = []
    cutoffs = np.percentile(
        df_map_count['usubjid'], [10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100])
    num_colors = len(cutoffs)
    colors = idw.interpolate_colors(
        ['FF3500', 'FF7400', 'FFBE00', 'A7FA00', '00EA66', '0000FF'],
        num_colors)

    # max_value = num_colors  # Assuming the max value is 10 for demonstration
    custom_scale = []
    for i, cutoff in enumerate(cutoffs):
        value_start = (
            0 if i == 0 else (cutoffs[i - 1] / max(df_map_count['usubjid'])))
        value_end = cutoff / max(df_map_count['usubjid'])
        color = colors[i]
        custom_scale.append([value_start, color])
        custom_scale.append([value_end, color])

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=df_map_count['country_iso'],
        z=df_map_count['usubjid'],
        text=df_map_count['slider_country'],
        colorscale=custom_scale,
        zmin=1,
        zmax=df_map_count['usubjid'].max(),
        marker_opacity=0.5,
        marker_line_width=0,
    ))

    mapbox_style = ['open-street-map', 'carto-positron']

    fig.update_layout(
        mapbox_style=mapbox_style[1],  # default
        mapbox_zoom=1.7,
        mapbox_center={'lat': 6, 'lon': -75},
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
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
def update_country_selection(
        select_all_value, selected_countries, all_countries_options):
    ctx = dash.callback_context

    if not ctx.triggered:
        # Initial load, no input has triggered the callback yet
        output = [
            selected_countries,
            [{'label': 'Unselect all', 'value': 'all'}], ['all']]

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'country-selectall':
        if 'all' in select_all_value:
            # 'Select all' (now 'Unselect all') is checked
            output = [
                [option['value'] for option in all_countries_options],
                [{'label': 'Unselect all', 'value': 'all'}], ['all']]
        else:
            # 'Unselect all' is unchecked
            output = [[], [{'label': 'Select all', 'value': 'all'}], []]
    elif trigger_id == 'country-checkboxes':
        if len(selected_countries) == len(all_countries_options):
            # All countries are selected manually
            output = [
                selected_countries,
                [{'label': 'Unselect all', 'value': 'all'}], ['all']]
        else:
            # Some countries are deselected
            output = [
                selected_countries,
                [{'label': 'Select all', 'value': 'all'}], []]
    else:
        output = [
            selected_countries,
            [{'label': 'Select all', 'value': 'all'}], select_all_value]
    return output


@app.callback(
    Output('country-fade', 'is_in'),
    [Input('country-display', 'n_clicks')],
    [State('country-fade', 'is_in')]
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
        return 'Country:'

    # Create a dictionary to map values to labels
    value_label_map = {
        option['value']: option['label'] for option in all_options}

    # Build the display string
    selected_labels = [
        value_label_map[val] for val in selected_values
        if val in value_label_map]
    display_text = ', '.join(selected_labels)

    if len(display_text) > 20:  # Adjust character limit as needed
        output = f'Country: {selected_labels[0]}, '
        output += f'+{len(selected_labels) - 1} more...'
    else:
        output = f'Country: {display_text}'
    return output


if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port='8080')
    app.run_server(debug=True)
