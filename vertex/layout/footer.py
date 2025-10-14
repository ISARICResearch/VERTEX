from dash import html

isaric_logo = "ISARIC_logo.png"
partners_logo_list = ["FIOCRUZ_logo.png", "gh.png", "puc_rio.png"]
funders_logo_list = ["wellcome-logo.png", "billmelinda-logo.png", "uk-international-logo.png", "FundedbytheEU.png"]

logo_style = {"height": "5vh", "margin": "2px 10px"}

footer = html.Div(
    [
        html.Img(src="assets/logos/" + isaric_logo, className="img-fluid", style={"height": "7vh", "margin": "2px 10px"}),
        html.P("In partnership with: ", style={"display": "inline"}),
    ]
    + [html.Img(src="assets/logos/" + logo, className="img-fluid", style=logo_style) for logo in partners_logo_list]
    + [html.P("    With funding from: ", style={"display": "inline"})]
    + [html.Img(src="assets/logos/" + logo, className="img-fluid", style=logo_style) for logo in funders_logo_list],
    style={
        "position": "absolute",
        "bottom": 0,
        "width": "calc(100% - 300px)",
        "margin-left": "300px",
        "background-color": "#FFFFFF",
        "zIndex": 50,
    },
)
