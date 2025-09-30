from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd


def define_filters_controls(
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

def get_filter_options(df_map):
    max_age = max((100, df_map['demog_age'].max()))
    age_options = {
        'min': 0,
        'max': max_age,
        'step': 10,
        'marks': {i: {'label': str(i)} for i in range(0, max_age + 1, 10)},
        'value': [0, max_age]
    }

    admdate_yyyymm = pd.date_range(
        start=df_map['pres_date'].min(),
        end=df_map['pres_date'].max(),
        freq='MS'
    )
    admdate_options = {
        'min': 0,
        'max': len(admdate_yyyymm) - 1,
        'step': 1,
        'marks': {i: {'label': d.strftime('%Y-%m')} for i, d in enumerate(admdate_yyyymm)},
        'value': [0, len(admdate_yyyymm) - 1]
    }

    outcome_options = [
        {'label': v, 'value': v}
        for v in df_map['filters_outcome'].dropna().unique()
    ]

    country_options = [
        {'label': c, 'value': c}
        for c in sorted(df_map['filters_country'].dropna().unique())
    ]

    sex_options = [
        {'label': 'Male', 'value': 'Male'},
        {'label': 'Female', 'value': 'Female'},
        {'label': 'Other / Unknown', 'value': 'Other / Unknown'}
    ]

    return {
        'sex_options': sex_options,
        'age_options': age_options,
        'admdate_options': admdate_options,
        'country_options': country_options,
        'outcome_options': outcome_options
    }
