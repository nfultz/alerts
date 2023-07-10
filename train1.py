import analysis
import pandas as pd
import xgboost as xgb

params = pd.read_pickle(analysis.HP_PICKLE)

out = analysis.objective(params)

pd.to_pickle(out, analysis.OBJ_PICKLE)