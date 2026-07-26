import pandas as pd
import numpy as np 
from scenario_engine004 import aggregate_summary
from scenario_engine004 import run_product_scenario

def run_scenario_report(d, changes, products=None, price_booster=None, feature_cols=None,
                          cat_cols=None, product_vol_stats=None, scaling_ref=None,
                          src_cols=None, imp_cols=None):
    """
    changes: dict, e.g. {'india': 0.7} or {'local': 1.2} or {'india': 0.7, 'local': 1.2}
    products: list of product names, or None to run on ALL products in d
    """
    target_products = products if products is not None else d['product_name'].unique()

    rows = []
    for p in target_products:
        result = run_product_scenario(d, p, changes, price_booster, feature_cols,
                                        cat_cols, product_vol_stats, scaling_ref, src_cols, imp_cols)
        rows.append(result)

    results_df = pd.DataFrame(rows)
    summary = aggregate_summary(results_df)
    return results_df, summary