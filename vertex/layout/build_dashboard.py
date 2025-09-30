from config_loader import get_config, load_vertex_data
from insight_panels import get_insight_panels  
from layout.layout_builder import define_app_layout

def build_dashboard(project_path: str, app) -> html.Div:
    config_dict = get_config(project_path)
    
    insight_panels_path = os.path.join(project_path, config_dict['insight_panels_path'])
    insight_panels, buttons = get_insight_panels(config_dict, insight_panels_path)

    df_map, df_forms_dict, dictionary, quality_report = load_vertex_data(project_path, config_dict)

    # Filters
    from your_module import merge_data_with_countries, get_countries, create_map  # adjust path

    df_map_with_countries = merge_data_with_countries(df_map)
    df_countries = get_countries(df_map_with_countries)

    filter_columns_dict = {
        'subjid': 'subjid',
        'demog_sex': 'filters_sex',
        'demog_age': 'filters_age',
        'dates_admdate': 'filters_admdate',
        'country_iso': 'filters_country',
        'outco_binary_outcome': 'filters_outcome'
    }

    df_filters = df_map_with_countries[filter_columns_dict.keys()].rename(columns=filter_columns_dict)
    df_map = pd.merge(df_map_with_countries, df_filters, on='subjid', how='left')
    df_forms_dict = {
        form: pd.merge(df_form, df_filters, on='subjid', how='left')
        for form, df_form in df_forms_dict.items()
    }

    map_layout_dict = dict(
        map_style='carto-positron',
        map_zoom=config_dict['map_layout_zoom'],
        map_center={
            'lat': config_dict['map_layout_center_latitude'],
            'lon': config_dict['map_layout_center_longitude']
        },
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
    )

    fig = create_map(df_countries, map_layout_dict)

    sex_options = [
        {'label': 'Male', 'value': 'Male'},
        {'label': 'Female', 'value': 'Female'},
        {'label': 'Other / Unknown', 'value': 'Other / Unknown'}
    ]

    max_age = max((100, df_map['demog_age'].max()))
    age_options = {'min': 0, 'max': max_age, 'step': 10}
    age_range = range(age_options['min'], age_options['max'] + 1, age_options['step'])
    age_options['marks'] = {ii: {'label': str(ii)} for ii in age_range}
    age_options['value'] = [age_options['min'], age_options['max']]

    admdate_yyyymm = pd.date_range(
        start=df_map['dates_admdate'].min().strftime('%Y-%m'),
        end=(df_map['dates_admdate'].max() + pd.DateOffset(months=1)).strftime('%Y-%m'),
        freq='MS'
    )
    admdate_yyyymm = [x.strftime('%Y-%m') for x in admdate_yyyymm]
    admdate_options = {
        'min': 0,
        'max': len(admdate_yyyymm) - 1,
        'step': 1,
        'marks': {i: {'label': d} for i, d in enumerate(admdate_yyyymm)},
        'value': [0, len(admdate_yyyymm) - 1]
    }

    country_options = [{'label': r['country_name'], 'value': r['country_iso']} for _, r in df_countries.iterrows()]
    outcome_options = [{'label': v, 'value': v} for v in df_map['filters_outcome'].dropna().unique()]

    filter_options = {
        'sex_options': sex_options,
        'age_options': age_options,
        'country_options': country_options,
        'admdate_options': admdate_options,
        'outcome_options': outcome_options,
    }

    layout = define_app_layout(fig, buttons, filter_options, map_layout_dict, config_dict['project_name'])

    # Optional: re-register callbacks if they depend on df_map etc.
    from callbacks import register_callbacks
    register_callbacks(
        app, insight_panels, df_map,
        df_forms_dict, dictionary, quality_report, filter_options,
        project_path, config_dict['save_filtered_public_outputs']
    )

    return layout
