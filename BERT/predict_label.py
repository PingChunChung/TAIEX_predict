from pathlib import Path
import pandas as pd
import numpy as np
import predict
from scipy.special import softmax


# +
pre_folder = 'Dcard_csv(data)'
after_folder = 'Dcard_csv(data)'

data_path = Path('data')
data_name = [ data.name for data in (data_path/'raw_data'/f'{pre_folder}').glob('*.csv')]
done_data_name = [ data.name for data in (data_path/'predicted_data'/f'{after_folder}').glob('*.csv')]
Path.mkdir(data_path/'predicted_data'/f'{after_folder}',parents=True, exist_ok=True)
# -

for idx,data in enumerate(data_name):
    try:
        if data not in done_data_name:
            print(data+'\n')
            print(f'{idx}/{len(data_name)}')
            predict_df = pd.read_csv(data_path/"raw_data"/f"{pre_folder}"/data, header = None, quoting=3, quotechar='"')
            listTestData = predict_df[0].dropna().tolist()

            predictions, raw_outputs = predict.predict(listTestData)
            probabilities = softmax(raw_outputs, axis=1)
            prob = [probabilities[idx][label] for idx,label in enumerate(predictions)] # 信心值

            predicted_df = pd.concat([predict_df, pd.Series(predictions), pd.Series(list(raw_outputs)), pd.Series(prob)],axis=1)
            predicted_df.columns = ['text','label','raw_outputs','probability']
            predicted_df.to_csv(f"{data_path}/predicted_data/{after_folder}/{data}", index = False)
    except:
        continue



