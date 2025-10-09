# modals.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from vertex.layout.filters import define_filters_controls

login_modal = dbc.Modal(
    id="login-modal",
    is_open=False,
    centered=True,
    backdrop="static",
    keyboard=False,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Login")),
        dbc.ModalBody([
            html.Form([
                dcc.Input(id="username", type="text", placeholder="Username"),
                html.Br(),
                dcc.Input(id="password", type="password", placeholder="Password"),
                html.Br(),
                html.Button("Submit", id="login-submit", n_clicks=0, type="button"),
                html.Div(id="login-output", style={"color": "red", "margin-top": "10px"}),
            ]),
            html.Div([
                html.Button("Register", id="open-register", n_clicks=0, className="btn btn-link", type="button"),
                html.Div(id="register-launcher-output")
            ])
        ]),
    ]
)

register_modal = dbc.Modal(
    id="register-modal",
    is_open=False,
    centered=True,
    backdrop="static",
    keyboard=False,
    children=[
        dbc.ModalHeader(dbc.ModalTitle("Register")),
        dbc.ModalBody([
            html.Form([
                dcc.Input(id="register-email", type="email", placeholder="Email"),
                html.Br(),
                dcc.Input(id="register-password", type="password", placeholder="Password"),
                html.Br(),
                dcc.Input(id="register-confirm-password", type="password", placeholder="Confirm Password"),
                html.Br(),
                html.Button("Register", id="register-submit", n_clicks=0, type="button"),
                html.Div(id="register-output", style={"color": "red", "margin-top": "10px"}),
            ])
        ])
    ]
)

#######################
# Insight Panel Modals
#######################

def create_modal(visuals, button, filter_options):
    if visuals is None:
        insight_children = []
        about_str = ''
    else:
        insight_children = [
            dbc.Tabs([
                dbc.Tab(dbc.Row(
                    [dbc.Col(dcc.Graph(figure=figure), id=id)]), label=label)
                for figure, id, label, _ in visuals], active_tab='tab-0')]
        # This text appears after clicking the insight panel's About button
        about_list = ['Information about each visual in the insight panel:']
        about_list += [
            '<strong>' + label + '</strong>' + about
            for _, _, label, about in visuals]
        about_str = '\n'.join(about_list)

    try:
        title = button['item'] + ': ' + button['label']
    except Exception:
        title = ''

    instructions_str = open('assets/instructions.txt', 'r').read()
    filters_modal = define_filters_controls(**filter_options, layout="modal", with_submit=True, prefix="modal")    
    modal = [
        dbc.ModalHeader(html.H3(
            title,
            id='line-graph-modal-title',
            style={'fontSize': '2vmin', 'fontWeight': 'bold'})
        ),
        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title='Filters and Controls',
                    children=[
                        filters_modal
                    ]),
                dbc.AccordionItem(
                    title='Insights', children=insight_children)
                ], active_item='item-1')
            ], style={
                'overflowY': 'auto', 'minHeight': '75vh', 'maxHeight': '75vh'}
        ),
        define_footer_modal(
            generate_html_text(instructions_str),
            generate_html_text(about_str))
    ]
    return modal

# Footer
def define_footer_modal(instructions, about):
    footer = dbc.ModalFooter([
        html.Div([
            dbc.Button(
                'About',
                id='modal_about_popover',
                color='info', size='sm', style={'margin-right': '5px'}),
            dbc.Button(
                'Instructions',
                id='modal_instruction_popover',
                size='sm', style={'margin-right': '5px'}),
        ], className='ml-auto'),
        dbc.Popover(
            [
                dbc.PopoverHeader(
                    'Instructions',
                    style={'fontWeight': 'bold'}),
                dbc.PopoverBody(instructions)
            ],
            target='modal_instruction_popover',
            trigger='hover',
            placement='top',
            hide_arrow=False,
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader('About', style={'fontWeight': 'bold'}),
                dbc.PopoverBody(about),
            ],
            target='modal_about_popover',
            trigger='hover',
            placement='top',
            hide_arrow=False,
        ),
    ])
    return footer

def generate_html_text(text):
    text_list = text.strip('\n').split('\n')
    div_list = []
    for line in text_list:
        strong_list = line.split('<strong>')
        for string in strong_list:
            if '</strong>' in string:
                strong, not_strong = string.split('</strong>')
                div_list.append(html.Div(html.Strong(strong)))
                div_list.append(html.Div(not_strong))
            else:
                div_list.append(html.Div(string))
        div_list.append(html.Br())
    div = html.Div(div_list[:-1])
    return div
