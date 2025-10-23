import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html


def define_filters_controls(
    sex_options,
    age_options,
    country_options,
    admdate_options,
    outcome_options,
    layout="accordion",
    with_submit=False,
    prefix="",
):
    """
    Unified filter component generator.

    Args:
        layout: "accordion" (default) or "modal".
        with_submit: include a submit button.
        prefix: optional string to prefix component IDs (e.g., "modal-").
        add_row: optional dbc.Row() to insert before submit button.
    """

    def cid(base):
        return f"{base}-{prefix}" if prefix else base

    sex_checklist = dcc.Checklist(
        id=cid("sex-checkboxes"),
        options=sex_options,
        value=[opt["value"] for opt in sex_options],
        inputStyle={"margin-right": "2px", "margin-left": "10px"},
        inline=(layout == "accordion"),
        style={"margin-left": "-10px"} if layout == "accordion" else {},
    )

    age_slider = dcc.RangeSlider(
        id=cid("age-slider"),
        min=age_options["min"],
        max=age_options["max"],
        step=age_options["step"],
        marks=age_options["marks"],
        value=age_options["value"],
        pushable=10,
    )

    admdate_slider = dcc.RangeSlider(
        id=cid("admdate-slider"),
        min=admdate_options["min"],
        max=admdate_options["max"],
        step=admdate_options["step"],
        marks=admdate_options["marks"],
        value=admdate_options["value"],
        pushable=1,
    )

    outcome_checklist = dcc.Checklist(
        id=cid("outcome-checkboxes"),
        options=outcome_options,
        value=[opt["value"] for opt in outcome_options],
        inputStyle={"margin-right": "2px", "margin-left": "10px"},
        inline=(layout == "accordion"),
        style={"margin-left": "-10px"} if layout == "accordion" else {},
    )

    country_section = html.Div(
        [
            html.Div(id=cid("country-display"), children=html.B("Country:"), style={"cursor": "pointer"}),
            dbc.Fade(
                html.Div(
                    [
                        dcc.Checklist(
                            id=cid("country-selectall"),
                            options=[{"label": "Select all", "value": "all"}],
                            value=["all"],
                            inputStyle={"margin-right": "2px"},
                        ),
                        dcc.Checklist(
                            id=cid("country-checkboxes"),
                            options=country_options,
                            value=[opt["value"] for opt in country_options],
                            style={"overflowY": "auto", "maxHeight": "100px"},
                            inputStyle={"margin-right": "2px"},
                        ),
                    ]
                ),
                id=cid("country-fade"),
                is_in=True,
                appear=True,
            ),
        ]
    )

    # layout differences
    if layout == "accordion":
        children = [
            html.Label(html.B("Sex at birth:")),
            html.Div(style={'margin-top': '5px'}),
            sex_checklist,
            html.Div(style={'margin-top': '10px'}),
            html.Label(html.B("Age:")),
            html.Div(style={'margin-top': '5px'}),
            age_slider,
            html.Div(style={'margin-top': '10px'}),
            country_section,
            html.Div(style={'margin-top': '15px'}),
            html.Label(html.B("Admission date:")),
            html.Div(style={'margin-top': '5px'}),
            admdate_slider,
            html.Div(style={'margin-top': '35px'}),
            html.Label(html.B("Outcome:")),
            html.Div(style={'margin-top': '5px'}),
            outcome_checklist,
        ]
        comp = dbc.AccordionItem(
            title="Filters and Controls", children=children, style={"overflowY": "auto", "maxHeight": "75vh"}
        )
    else:
        # modal layout in rows
        rows = [
            dbc.Row(
                [
                    dbc.Col([html.H6("Sex at birth:"), sex_checklist], width=2),
                    dbc.Col([html.H6("Age:"), age_slider], width=3),
                    dbc.Col([html.H6("Admission date:"), admdate_slider], width=3),
                    dbc.Col([html.H6("Country:"), country_section], width=2),
                    dbc.Col([html.H6("Outcome:"), outcome_checklist], width=2),
                ]
            )
        ]
        if with_submit:
            rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [dbc.Button("Submit", id=cid("submit-button"), color="primary", className="mr-2")],
                            width={"size": 6, "offset": 3},
                            style={"textAlign": "center"},
                        )
                    ]
                )
            )
        comp = dbc.Row([dbc.Col(rows)])

    return comp


def define_filters_controls_modal(
    sex_options,
    age_options,
    country_options,
    admdate_options,  # disease_options,
    outcome_options,
    add_row=None,
):
    filter_rows = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6("Sex at birth:", style={"margin-right": "10px"}),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="sex-checkboxes-modal",
                                    options=sex_options,
                                    value=[option["value"] for option in sex_options],
                                    inputStyle={"margin-right": "2px"},
                                )
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6("Age:", style={"margin-right": "10px"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.RangeSlider(
                                            id="age-slider-modal",
                                            min=age_options["min"],
                                            max=age_options["max"],
                                            step=age_options["step"],
                                            marks=age_options["marks"],
                                            value=age_options["value"],
                                        )
                                    ],
                                    style={"width": "100%"},
                                )  # Apply style to this div
                            ]
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H6("Admission date:", style={"margin-right": "10px"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.RangeSlider(
                                            id="admdate-slider-modal",
                                            min=admdate_options["min"],
                                            max=admdate_options["max"],
                                            step=admdate_options["step"],
                                            marks=admdate_options["marks"],
                                            value=admdate_options["value"],
                                        )
                                    ],
                                    style={"width": "100%"},
                                )  # Apply style to this div
                            ]
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H6("Country:", style={"margin-right": "10px"}),
                        html.Div(
                            [
                                html.Div(id="country-display-modal", children="Country:", style={"cursor": "pointer"}),
                                dbc.Fade(
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="country-selectall-modal",
                                                options=[{"label": "Select all", "value": "all"}],
                                                value=["all"],
                                                inputStyle={"margin-right": "2px"},
                                            ),
                                            dcc.Checklist(
                                                id="country-checkboxes-modal",
                                                options=country_options,
                                                value=[option["value"] for option in country_options],
                                                style={"overflowY": "auto", "maxHeight": "100px"},
                                                inputStyle={"margin-right": "2px"},
                                            ),
                                        ]
                                    ),
                                    id="country-fade-modal",
                                    is_in=True,
                                    appear=True,
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6("Outcome:", style={"margin-right": "10px"}),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="outcome-checkboxes-modal",
                                    options=outcome_options,
                                    value=[option["value"] for option in outcome_options],
                                    inputStyle={"margin-right": "2px"},
                                )
                            ]
                        ),
                    ],
                    width=2,
                ),
            ]
        )
    ]
    row_button = dbc.Row(
        [
            dbc.Col(
                [dbc.Button("Submit", id="submit-button-modal", color="primary", className="mr-2")],
                width={"size": 6, "offset": 3},
                style={"text-align": "center"},
            )  # Center the button
        ]
    )
    row_list = filter_rows + [row_button]
    if add_row is not None:
        row_list = filter_rows + [add_row, row_button]
    filters = dbc.Row([dbc.Col(row_list)])
    return filters


def get_filter_options(df_map):
    marks_style = {
        "text-align": "right",
        "transform-origin": "bottom left",
        "transform": "rotate(-45deg)",
        "margin-left": "-5px",
        "margin-top": "25px",
        "height": "70px",
        "width": "70px",
    }

    max_age = max((100, df_map["demog_age"].max()))
    age_options = {
        "min": 0,
        "max": max_age,
        "step": 10,
        "marks": {i: {"label": str(i), "style": marks_style} for i in range(0, max_age + 1, 10)},
        "value": [0, max_age],
    }

    admdate_yyyymm = pd.date_range(start=df_map["pres_date"].min(), end=df_map["pres_date"].max() + pd.DateOffset(months=1), freq="MS")
    admdate_options = {
        "min": 0,
        "max": len(admdate_yyyymm) - 1,
        "step": 1,
        "marks": {i: {"label": d.strftime("%Y-%m"), "style": marks_style} for i, d in enumerate(admdate_yyyymm)},
        "value": [0, len(admdate_yyyymm) - 1],
    }

    outcome_options = [{"label": v, "value": v} for v in df_map["filters_outcome"].dropna().unique()]

    country_options = [{"label": c, "value": c} for c in sorted(df_map["filters_country"].dropna().unique())]

    sex_options = [
        {"label": "Male", "value": "Male"},
        {"label": "Female", "value": "Female"},
        {"label": "Other / Unknown", "value": "Other / Unknown"},
    ]

    return {
        "sex_options": sex_options,
        "age_options": age_options,
        "admdate_options": admdate_options,
        "country_options": country_options,
        "outcome_options": outcome_options,
    }
