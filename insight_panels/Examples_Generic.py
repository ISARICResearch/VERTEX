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
    button_item = 'Examples'
    button_label = 'Generic Panel'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''
    # Provide a list of all ARC data sections needed in the insight panel
    # Only variables from these sections will appear in the visuals
    sections = [
        'dates',  # Onset & presentation (REQUIRED)
        'demog',  # Demographics (REQUIRED)
        'daily',  # Daily sections (REQUIRED)
        'asses',  # Assessment (REQUIRED)
        'outco',  # Outcome (REQUIRED)
        'inclu',  # Inclusion criteria
        'readm',  # Re-admission and previous pin
        'travel',  # Travel history
        'expo14',  # Exposure history in previous 14 days
        'preg',  # Pregnancy
        'infa',  # Infant
        'comor',  # Co-morbidities and risk factors
        'medic',  # Medical history
        'drug7',  # Medication previous 7-days
        'drug14',  # Medication previous 14-days
        'vacci',  # Vaccination
        'advital',  # Vital signs & assessments on admission
        'adsym',  # Signs and symptoms on admission
        'vital',  # Vital signs & assessments
        'sympt',  # Signs and symptoms
        'lesion',  # Skin & mucosa assessment
        'treat',  # Treatments & interventions
        'labs',  # Laboratory results
        'imagi',  # Imaging
        'medi',  # Medication
        'test',  # Pathogen testing
        'diagn',  # Diagnosis
        'compl',  # Complications
        'inter',  # Interventions
        'follow',  # Follow-up assessment
        'withd',  # Withdrawal
    ]
    variable_list = ['subjid']
    variable_list += [
        col for col in ia.get_variable_list(dictionary, sections)
        if col in df_map.columns]
    df_map = df_map[variable_list].copy()

    fig1 = idw.fig_placeholder(
        df_map,
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Figure 1',
        graph_about='Placeholder figure 1')
    return (fig1,)
