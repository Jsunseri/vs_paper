import pickle
import pandas as pd
import numpy as np
import sys

newmethods = ["dense", "crossdock_default2018", "general_default2018"]
# sdsorter output summary is
# Rank Title Vina MW Target File DENSE0 
# DENSE1 DENSE2 DENSE3 DENSE4 CROSS0 CROSS1 CROSS2 CROSS3 CROSS4 REF0 REF1 REF2
# REF3 REF4 GENERAL0 GENERAL1 GENERAL2 GENERAL3 GENERAL4
dirs = np.genfromtxt(sys.argv[1], dtype='str')
df = pd.DataFrame()
for dir in dirs:
    df = pd.concat([df, pd.read_csv("%s/sdsorter.summary" %dir,
        delim_whitespace=True, header=0)], ignore_index=True)
    
# need to get dataframe with actual predictions, let's use the ensemble mean
# over the highest predicted score as before
df.loc[:,'Vina'] = df[['Vina']].mul(-1) # sigh

df["label"] = [0 if "inactive" in item else 1 for item in df["File"]]
pickle.dump(df, open('preds_df.cpickle', 'wb'), -1)
