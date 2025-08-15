import dash_html_components as html
from dash import dcc
import dash_bootstrap_components as dbc


def define_filters_and_controls(
        sex_options, age_options, country_options,
        admdate_options,  # disease_options,
        outcome_options):
    filters = dbc.AccordionItem(
        title='Filters and Controls',
        children=[
            html.Label(html.B('Sex at birth:')),
            html.Div(style={'margin-top': '5px'}),
            dcc.Checklist(
                id='sex-checkboxes',
                options=sex_options,
                value=[option['value'] for option in sex_options],
                inputStyle={'margin-right': '2px', 'margin-left': '10px'},
                style={'margin-left': '-10px'},
                inline=True
            ),
            html.Div(style={'margin-top': '10px'}),
            html.Label(html.B('Age:')),
            html.Div(style={'margin-top': '5px'}),
            dcc.RangeSlider(
                id='age-slider',
                min=age_options['min'],
                max=age_options['max'],
                step=age_options['step'],
                marks=age_options['marks'],
                value=age_options['value'],
                pushable=10,
            ),
            html.Div(style={'margin-top': '10px'}),
            html.Div([
                html.Div(
                    id='country-display',
                    children=html.Div([
                        html.B('Country:'),
                        # ' (scroll down for all)'
                    ]),
                    style={'cursor': 'pointer'}),
                dbc.Fade(
                    html.Div([
                        dcc.Checklist(
                            id='country-selectall',
                            options=[{
                                'label': 'Select all',
                                'value': 'all'
                            }],
                            value=['all'],
                            inputStyle={'margin-right': '2px'}
                        ),
                        dcc.Checklist(
                            id='country-checkboxes',
                            options=country_options,
                            value=[
                                option['value'] for option in country_options],
                            style={
                                'overflowY': 'auto',
                                'maxHeight': '70px'
                            },
                            inputStyle={'margin-right': '2px'}
                        )
                    ]),
                    id='country-fade',
                    is_in=True,
                    appear=True,
                )
            ]),
            html.Div(style={'margin-top': '15px'}),
            html.Label(html.B('Admission date:')),
            html.Div(style={'margin-top': '5px'}),
            dcc.RangeSlider(
                id='admdate-slider',
                min=admdate_options['min'],
                max=admdate_options['max'],
                step=admdate_options['step'],
                marks=admdate_options['marks'],
                value=admdate_options['value'],
                pushable=1,
            ),
            html.Div(style={'margin-top': '35px'}),
            html.Label(html.B('Outcome:')),
            html.Div(style={'margin-top': '5px'}),
            dcc.Checklist(
                id='outcome-checkboxes',
                options=outcome_options,
                value=[option['value'] for option in outcome_options],
                inputStyle={'margin-right': '2px', 'margin-left': '10px'},
                style={'margin-left': '-10px'},
                inline=True
            )
        ], style={'overflowY': 'auto', 'maxHeight': '75vh'},
    )
    return filters

