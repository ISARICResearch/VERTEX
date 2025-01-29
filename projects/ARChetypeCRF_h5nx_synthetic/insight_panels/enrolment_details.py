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
    button_item = 'Enrolment'
    button_label = 'Enrolment Details'
    output = {'item': button_item, 'label': button_label}
    return output


def create_visuals(
        df_map, df_forms_dict, dictionary, quality_report,
        filepath, suffix, save_inputs):
    '''
    Create all visuals in the insight panel from the RAP dataframe
    '''

    df_sunburst = df_map[['subjid', 'site', 'filters_country']].groupby(
        ['site', 'filters_country']).nunique().reset_index()
    df_sunburst['site'] = df_sunburst['site'].str.split('-', n=1).str[0]

    df_enrolment = df_map[
        ['subjid', 'filters_country', 'dates_enrolmentdate']].copy()
    groupby_columns = ['filters_country', 'dates_enrolmentdate']
    df_enrolment = df_enrolment[['subjid'] + groupby_columns].groupby(
        groupby_columns).nunique().reset_index()
    df_enrolment.rename(columns={
        'filters_country': 'stack_group',
        'dates_enrolmentdate': 'timepoint',
        'subjid': 'value'}, inplace=True)

    # Convert 'timepoint' to datetime format and sort
    df_enrolment['timepoint'] = pd.to_datetime(
        df_enrolment['timepoint'], errors='coerce')
    df_enrolment = df_enrolment.pivot_table(
        index='timepoint', columns='stack_group',
        values='value', aggfunc='sum').fillna(0)

    df_enrolment = df_enrolment.reset_index()
    df_enrolment = df_enrolment.sort_values('timepoint')
    df_enrolment['timepoint'] = df_enrolment['timepoint'].apply(
        lambda x: x.strftime('%m-%Y'))
    df_enrolment.rename(columns={'timepoint': 'index'}, inplace=True)

    ######################################
    ######################################
    # Cumulative chart
    ######################################

    df_cumulative = df_enrolment.copy()
    columns = [col for col in df_cumulative.columns if col != 'index']
    df_cumulative[columns] = df_cumulative[columns].cumsum()

    #########################################
    #########################################

    fig_patients_bysite = idw.fig_sunburst(
        df_sunburst,
        title='Enrolment by site (SYNTHETIC DATA)',
        path=['filters_country', 'site'], values='subjid',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Site Enrolment', graph_about='...')
    fig_cumulative = idw.fig_stacked_bar_chart(
        df_cumulative,
        title='Cumulative enrolment (SYNTHETIC DATA)',
        xlabel='Month', ylabel='Number of patients',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Cumulative patient enrolment by country',
        graph_about='...')
    fig_enrolment = idw.fig_stacked_bar_chart(
        df_enrolment,
        title='Enrolment by month (SYNTHETIC DATA)',
        xlabel='Month', ylabel='Number of patients',
        suffix=suffix, filepath=filepath, save_inputs=save_inputs,
        graph_label='Patient enrolment by country', graph_about='...')

    return (fig_enrolment, fig_cumulative, fig_patients_bysite)
