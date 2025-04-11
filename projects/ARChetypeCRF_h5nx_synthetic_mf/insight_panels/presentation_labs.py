import numpy as np
import pandas as pd
import IsaricDraw as idw
import IsaricAnalytics as ia


def define_button():
    '''Defines the button in the main dashboard menu'''
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = 'Clinical Presentation'
    button_label = 'Lab Results'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    # Demographics and comorbidities descriptive table
    # split_column = 'demog_sex'
    # split_column_order = ['Female', 'Male', 'Other / Unknown']
    split_column = 'outco_binary_outcome'
    split_column_order = ['Discharged', 'Death', 'Censored']
    sections = ['labs']
    df_table = ia.get_descriptive_data(
        df_map, dictionary, by_column=split_column,
        include_sections=sections, exclude_negatives=False)
    table, table_key = ia.descriptive_table(
        df_table, dictionary, by_column=split_column,
        column_reorder=split_column_order)
    fig_table = idw.fig_table(
        table, table_key=table_key,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Descriptive Table*',
        graph_about='Summary of demographics and comorbidities.')

    disclaimer_text = '''Disclaimer: the underlying data for these figures is \
synthetic data. Results may not be clinically relevant or accurate.'''
    disclaimer_df = pd.DataFrame(
        disclaimer_text, columns=['paragraphs'], index=range(1))
    disclaimer = idw.fig_text(
        disclaimer_df,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='*DISCLAIMER: SYNTHETIC DATA*',
        graph_about=disclaimer_text
    )

    return (fig_table, disclaimer)
