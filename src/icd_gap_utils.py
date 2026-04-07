from __future__ import annotations

import pandas as pd


def add_gap_entries(
    df: pd.DataFrame,
    hadm_id_val: list[int] | None = None,
    gap_list: list[int] = [30, 10, 5, 1],
) -> pd.DataFrame:
    """Insert GAP_<n> token rows between ICD procedure entries for a given admission.

    For each consecutive pair of rows within an admission, if the time difference
    exceeds a gap threshold, a synthetic GAP_<n> row is inserted. Gaps are filled
    greedily using the largest fitting threshold first (recursive).

    Args:
        df: DataFrame containing procedure ICD data for one or more admissions.
        hadm_id_val: List of hadm_ids to process. If None, processes all hadm_ids.
        gap_list: Descending list of gap thresholds in days. Defaults to [30, 10, 5, 1].

    Returns:
        DataFrame with gap rows inserted for the specified admission(s).
    """
    if hadm_id_val is None:
        return pd.concat([
            add_gap_entries(group, [hid], gap_list=gap_list)
            for hid, group in df.groupby('hadm_id')
        ]).reset_index(drop=True)

    subset = df[df['hadm_id'].isin(hadm_id_val)].copy()
    subset = subset.sort_values(by=['chartdate', 'seq_num'])
    gap_list = sorted(gap_list, reverse=True)

    final_rows = []
    current_data = subset.reset_index(drop=True)

    for i in range(len(current_data) - 1):
        current_row = current_data.iloc[i]
        next_row = current_data.iloc[i + 1]

        final_rows.append(current_row)

        time_diff = (next_row['chartdate'] - current_row['chartdate']).days

        def fill_gap(remaining_time, current_ref_time):
            for gap in gap_list:
                if remaining_time > gap:
                    gap_entry = current_row.copy()
                    gap_entry['icd_code'] = f'GAP_{gap}'
                    gap_entry['icd_code_mapped'] = f'GAP_{gap}'
                    gap_entry['chartdate'] = current_ref_time + pd.Timedelta(days=gap)
                    gap_entry['pos'] = 1
                    gap_entry['icd_version'] = 'gap'
                    final_rows.append(gap_entry)
                    fill_gap(remaining_time - gap, gap_entry['chartdate'])
                    return

        fill_gap(time_diff, current_row['chartdate'])

    if len(current_data) > 0:
        final_rows.append(current_data.iloc[-1])

    return pd.DataFrame(final_rows)



