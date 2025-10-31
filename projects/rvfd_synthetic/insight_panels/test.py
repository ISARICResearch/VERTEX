import vertex.IsaricDraw as idw
import vertex.IsaricAnalytics as ia


def define_button():
    '''Defines the button in the main dashboard menu'''
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = 'Initial insight panel'
    button_label = 'Everything'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    split_column = 'demog_sex'
    split_column_order = ['Femme', 'Homme']

    sections = ['pres', 'demog', 'comor', 'adsym', 'adsign', 'preg', 'vital', 'sympt', 'sign', 'treat', 'critd', 'labs', 'medi', 'outco', 'follow']

    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=sections)
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Descriptive Table',
        graph_about='Summary of all data.')

    return (fig_table, )
