from sklearn.linear_model import LogisticRegression, LinearRegression as SklearnLinearRegression
import numpy as np
import scipy.stats as stat
import pandas as pd 
import matplotlib.pylab as plt 

class LogisticRegression_with_p_values:
    
    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)
    
    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.original_feature_names = X.columns
            X_np = X.values
        else:
            self.original_feature_names = [f'x{i}' for i in range(X.shape[1])]
            X_np = X

        non_constant_columns = np.var(X_np, axis=0) != 0
        self.kept_mask = non_constant_columns
        self.kept_feature_names = np.array(self.original_feature_names)[non_constant_columns]
        X_clean = X_np[:, non_constant_columns]

        self.model.fit(X_clean, y)
        
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X_clean)))
        denom = np.tile(denom, (X_clean.shape[1], 1)).T
        F_ij = np.dot((X_clean / denom).T, X_clean).astype(float)

        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diag(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)

    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)

    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']

    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']

    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()

    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_n_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()

    return df

# hàm cho continuous không sort theo WoE vì muốn preserve the natural order
def woe_ordered_continuous(df, continuous_variable_name, good_bad_variable_df):
    df = pd.concat([df[continuous_variable_name], good_bad_variable_df], axis=1)

    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)

    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']

    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']

    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()

    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])

    #df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_n_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()

    return df

def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']

    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title('Weight of Evidence by ' + str(df_WoE.columns[0]))
    plt.xticks(rotation=rotation_of_x_axis_labels)
    plt.show()

def create_summary_table(model):
    summary_table = pd.DataFrame()
    summary_table['Feature name'] = model.kept_feature_names
    summary_table['Coefficients'] = model.coef_[0]
    summary_table['p-values'] = model.p_values

    intercept_row = pd.DataFrame({
        'Feature name': ['Intercept'],
        'Coefficients': [model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_],
        'p-values': [np.nan]
    })
    summary_table = pd.concat([intercept_row, summary_table], ignore_index=True)
    summary_table['p-values'] = summary_table['p-values'].apply(
        lambda x: f"{x:.6f}" if pd.notnull(x) else x
    )
    return summary_table
    
class LinearRegression(SklearnLinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, positive=False, n_jobs=1):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, positive=positive, n_jobs=n_jobs)
        self.normalize = normalize


    def fit(self, X, y, n_jobs=1):
        # Lưu tên feature
        if hasattr(X, 'columns'):
            self.kept_feature_names = X.columns.tolist()
            X_np = X.values
        else:
            self.kept_feature_names = [f"x{i}" for i in range(X.shape[1])]
            X_np = X

        super().fit(X_np, y, n_jobs)
        sse = np.sum((self.predict(X_np) - y) ** 2, axis=0) / float(X_np.shape[0] - X_np.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X_np.T, X_np))))])
        self.t = self.coef_ / se
        self.p_values = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X_np.shape[1])))
        return self
