import pandas as pd
from sklearn.utils import resample


def upsample_data(filename):
    df = pd.read_csv(filename)
    sample = df.sample(n=20)
    #print(sample)
    df.drop(sample.index,inplace=True)
    #print(len(df))
    df_majority = df[df["Job Zone"]==2]
    len_majority = len(df_majority)
    for i in range(1,6):
        if(i!=2):
            df_minority = df[df["Job Zone"]==i]
            df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=len_majority,    # to match majority class
                                     random_state=42)
            df_majority = pd.concat([df_majority, df_minority_upsampled])

    return df_majority,sample