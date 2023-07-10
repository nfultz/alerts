import analysis
import pandas as pd

Y, *R, _ = analysis.get_files()
ret = analysis.gen_features(*R)
pd.to_pickle( (Y, *ret), analysis.FEATURE_PICKLE)