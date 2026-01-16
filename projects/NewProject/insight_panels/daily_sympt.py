import numpy as np
import pandas as pd

import vertex.IsaricAnalytics as ia
import vertex.IsaricDraw as idw
import re

def define_button():
    """Defines the button in the main dashboard menu"""
    # Insight panels are grouped together by the button_item. Multiple insight
    # panels can share the same button_item are grouped in the dashboard menu
    # according to this
    # However, the combination of button_item and button_label must be unique
    button_item = "Daily Patient Observations"
    button_label = "Symptoms"
    output = {"item": button_item, "label": button_label}
    return output


def create_visuals(df_map, df_forms_dict, dictionary, quality_report, filepath, suffix, save_inputs):
    """
    Create all visuals in the insight panel from the RAP dataframe
    """
   
    #df_forms_dict
    daily_events=['Jour 1', 'Jour 3  (+1) ','Jour 7  (+2)',"Jour 14 (+/-2)",'Jour 28  (+/-5)','Jour 90  (+/-20)',"Jour 180 (+/-20)"]

    section='sympt'
    s=dictionary['field_name']
    vars_of_interest = s[s.str.startswith(section+"_", na=False)].tolist()

    #vars_of_interest=['vital_highesttem_c','vital_hr','vital_rr']
    df_long = ia.build_all_patients_event_dataframe(
        daily_forms_data=df_forms_dict,
        daily_events=daily_events,
        variables=vars_of_interest,   # list of variable names you want summarised
        patient_col="subjid",
        day_col="day",
    )

    df_table = ia.get_descriptive_data(
        df_long,
        dictionary,
        by_column="day",
        include_sections=[section],   # or whichever sections include your vars
        exclude_negatives=False
    )

    table_daily, table_key_daily = ia.descriptive_table(
        df_table,
        dictionary,
        by_column="day",
        column_reorder=daily_events
    )

    fig_table_daily= idw.fig_table(
        table_daily,
        table_key=table_key_daily,
        suffix=suffix,
        filepath=filepath,
        save_inputs=save_inputs,
        graph_label="Descriptive Table",
        graph_about="...",
    )
    

    def to_heatmap_df(
        df: pd.DataFrame,
        variable_col: str = "Variable",
        index_col_out: str = "index",
        index_column: str | None = None,
        drop_cols: tuple[str, ...] = ("All",),
        drop_header_like: bool = True,
        keep_nonmatching_as_is: bool = True,
        coerce_numeric: bool = True,
    ) -> pd.DataFrame:
        """
        Single function that:
        1) Extracts the leading count from strings like '12 (50.0) | 24' (optionally leaving non-matching cells as-is),
        2) Ensures a label column exists and renames it to `index_col_out`,
        3) Cleans HTML-ish labels (e.g., <b>...</b>, <i>...</i>) and removes trailing '(*)',
        4) Optionally drops header/section rows (Totals / SYMPT... / Saisir...),
        5) Drops unwanted columns (e.g., 'All'),
        6) Optionally coerces value columns to numeric.

        Output is compatible with fig_heatmaps(index_column=index_col_out).

        Parameters
        ----------
        df : pd.DataFrame
            Input table (often your descriptive_table output).
        variable_col : str
            Name of the column containing variable labels, if present.
            If not present, the function will reset_index() and use the index as labels.
        index_col_out : str
            Output column name to be used by fig_heatmaps as index_column (y-axis).
        index_column : str | None
            If provided, overrides index_col_out (kept for naming consistency).
        drop_cols : tuple[str, ...]
            Columns to drop before plotting (e.g., ('All',)).
        drop_header_like : bool
            Remove rows that look like section headers (Totals / SYMPT... / Saisir...).
        keep_nonmatching_as_is : bool
            If True: only transform cells matching the descriptive pattern; else set non-matching strings to NaN.
        coerce_numeric : bool
            If True: convert value columns to numeric with errors='coerce'.

        Returns
        -------
        pd.DataFrame
            Heatmap-ready dataframe: one label column + numeric value columns.
        """

        out = df.copy()
        if index_column is None:
            index_column = index_col_out

        # --- A) Ensure we have a variable label column
        if variable_col not in out.columns:
            out = out.reset_index().rename(columns={"index": variable_col})

        # --- B) Clean label text (strip HTML tags, normalize whitespace, drop trailing '(*)')
        def clean_label(x):
            if not isinstance(x, str):
                return x
            x = re.sub(r"<[^>]+>", "", x)          # remove HTML tags like <b>, <i>, etc.
            x = re.sub(r"\s*\(\*\)\s*$", "", x)    # remove trailing "(*)"
            x = re.sub(r"\s+", " ", x).strip()
            return x

        out[variable_col] = out[variable_col].map(clean_label)

        # --- C) Optionally drop header-like rows
        if drop_header_like:
            bad = (
                out[variable_col].str.contains(r"^Totals$", case=False, na=False)
                | out[variable_col].str.contains(r"SYMPT", case=False, na=False)
                | out[variable_col].str.contains(r"^Saisir", case=False, na=False)
            )
            out = out.loc[~bad].copy()

        # --- D) Drop unwanted columns (e.g., 'All')
        for c in drop_cols:
            if c in out.columns:
                out = out.drop(columns=c)

        # --- E) Extract leading counts from descriptive strings in value cells
        # Strict pattern for "count (something) | something"
        # If you want it looser, change it to r"^\s*(\d+)"
        desc_pattern = re.compile(r"^\s*(\d+)\s*\(.*?\)\s*\|.*$")

        def extract_cell(x):
            if isinstance(x, str):
                m = desc_pattern.match(x)
                if m:
                    return float(m.group(1))
                return x if keep_nonmatching_as_is else np.nan
            return x  # keep numeric/NaN/etc as-is

        # Apply extraction to all columns except the label column
        value_cols = [c for c in out.columns if c != variable_col]
        out[value_cols] = out[value_cols].applymap(extract_cell)

        # --- F) Rename label column to what fig_heatmaps expects
        out = out.rename(columns={variable_col: index_column})

        # --- G) Coerce values to numeric if desired
        if coerce_numeric:
            val_cols = [c for c in out.columns if c != index_column]
            for c in val_cols:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        return out
    

    daily_events=	['Jour 1', 'Jour 3  (+1) ','Jour 7  (+2)',"Jour 14 (+/-2)",'Jour 28  (+/-5)','Jour 90  (+/-20)',"Jour 180 (+/-20)"]


    DAY_COLUMN_MAP = {
        'Jour 1': " 1 ",
        'Jour 3  (+1) ': " 3 ",
        'Jour 7  (+2)': " 7 ",
        'Jour 14 (+/-2)': " 14",
        'Jour 28  (+/-5)': " 28",
        'Jour 90  (+/-20)': " 90",
        'Jour 180 (+/-20)': "180",
    }

    df_hm = to_heatmap_df(table_daily, variable_col="Variable", drop_cols=("All",))
    for col in daily_events:
        if col not in df_hm.columns:
            df_hm[col] = np.nan
    df_hm=df_hm[['index']+daily_events]
    df_hm=df_hm.fillna(0)
    df_hm = df_hm.rename(
        columns=lambda c: DAY_COLUMN_MAP.get(c, c)
    )
    fig_hm = idw.fig_heatmaps(
        df_hm,
        index_column="index",
        title="Symptoms over time",
        graph_label="..."
    )

    '''
    tdv=table_daily.copy()
    tdv.index=tdv['Variable']
    #tdv.drop(columns=['Variable'],inplace=True)
    tdv=extract_leading_count_df(tdv)
    tdv=tdv.fillna(0)
    tdv.drop(columns=['Variable'],inplace=True)
    tdv=tdv.reset_index()
    fig_hm= idw.fig_heatmaps(
        data=tdv,
        title="Vitals over time",
        xlabel="Day",
        ylabel="Variable",
        colorbar_label="Value",
        index_column="Variable",
        graph_label="...",
    )    '''

    return (fig_table_daily,fig_hm)
    #return (pyramid_chart, fig_table, freq_chart_comor, upset_plot_comor)
