
import numpy as np
from scipy.special import factorial

from sklearn.datasets import fetch_openml

# the following is from a sklearn page
def load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser="pandas").data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser="pandas").data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]

def poisson_likelihood(actu, pred):
    # the factorial is from the scipy package
    return (np.power(pred, actu)) * np.exp(-pred) / factorial(actu)

def poisson_loglikelihood(actu, pred):
    return np.log(poisson_likelihood(actu, pred))

def poisson_loglikelihood2(actu, pred):
    # just a demonstration to make sure my formulas work
    return actu * np.log(pred) - pred - np.log(factorial(actu))

