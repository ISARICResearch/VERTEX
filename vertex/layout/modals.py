# modals.py
import dash_bootstrap_components as dbc
from dash import dcc, html

from vertex.layout.filters import define_filters_controls
from vertex.logging.logger import setup_logger

logger = setup_logger(__name__)

#######################
# Insight Panel Modals
#######################


def create_modal(visuals, button, filter_options=None):
    if visuals is None:
        insight_children = []
        about_str = ""
    else:
        # NOTE(ADW): We keep all tab panes layout-measurable via CSS in assets/style.css.
        # This prevents plotly resize jumps when inactive tabs become visible.
        insight_children = [
            dbc.Tabs(
                [
                    dbc.Tab(
                        dbc.Row([dbc.Col(dcc.Graph(figure=figure, style={"width": "100%", "height": "100%"}), id=id)]),
                        label=label,
                    )
                    for figure, id, label, _ in visuals
                ],
                active_tab="tab-0",
            )
        ]
        logger.debug(f" Creating modal for button: {button} with {len(visuals)} visuals ")

        about_list = ["Information about each visual in the insight panel:"]
        about_list += ["<strong>" + label + "</strong>" + about for _, _, label, about in visuals]
        about_str = "\n".join(about_list)

    try:
        title = f"{button.get('item', '')}: {button.get('label', '')}"
    except Exception:
        title = ""

    instructions_str = open("assets/instructions.txt", "r").read()

    accordion_items = []

    # Only add filters if available
    if filter_options is not None:
        filters_modal = define_filters_controls(**filter_options, layout="modal", with_submit=True, prefix="modal")
        accordion_items.append(dbc.AccordionItem(title="Filters and Controls", item_id="filters", children=[filters_modal]))
    accordion_items.append(dbc.AccordionItem(title="Insights", item_id="insights", children=insight_children))

    modal = [
        dbc.ModalHeader(html.H3(title, id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),
        dbc.ModalBody(
            [dbc.Accordion(accordion_items, active_item="insights")],
            style={"overflowY": "auto", "minHeight": "75vh", "maxHeight": "75vh"},
        ),
        define_footer_modal(generate_html_text(instructions_str), generate_html_text(about_str)),
    ]
    return modal


# Footer
def define_footer_modal(instructions, about):
    footer = dbc.ModalFooter(
        [
            html.Div(
                [
                    dbc.Button("About", id="modal_about_popover", color="info", size="sm", style={"margin-right": "5px"}),
                    dbc.Button("Instructions", id="modal_instruction_popover", size="sm", style={"margin-right": "5px"}),
                ],
                className="ml-auto",
            ),
            dbc.Popover(
                [dbc.PopoverHeader("Instructions", style={"fontWeight": "bold"}), dbc.PopoverBody(instructions)],
                target="modal_instruction_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("About", style={"fontWeight": "bold"}),
                    dbc.PopoverBody(about),
                ],
                target="modal_about_popover",
                trigger="hover",
                placement="top",
                hide_arrow=False,
            ),
        ]
    )
    return footer


def generate_html_text(text):
    text_list = text.strip("\n").split("\n")
    div_list = []
    for line in text_list:
        strong_list = line.split("<strong>")
        for string in strong_list:
            if "</strong>" in string:
                strong, not_strong = string.split("</strong>")
                div_list.append(html.Div(html.Strong(strong)))
                div_list.append(html.Div(not_strong))
            else:
                div_list.append(html.Div(string))
        div_list.append(html.Br())
    div = html.Div(div_list[:-1])
    return div
