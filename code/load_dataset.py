import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold


def process_german(df, preprocess):
    df['status'] = df['status'].map({'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}).astype(int)
    df['credit_hist'] = df['credit_hist'].map({'A34': 0, 'A33': 1, 'A32': 2, 'A31': 3, 'A30': 4}).astype(int)

    df['savings'] = df['savings'].map({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}).astype(int)
    df['employment'] = df['employment'].map({'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}).astype(int)    
    df['gender'] = df['personal_status'].map({'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}).astype(int)
    df['debtors'] = df['debtors'].map({'A101': 0, 'A102': 1, 'A103': 2}).astype(int)
    df['property'] = df['property'].map({'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}).astype(int)        
    df['install_plans'] = df['install_plans'].map({'A141': 1, 'A142': 1, 'A143': 0}).astype(int)
    if preprocess:
        df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')],axis=1)
        df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
        df.loc[(df['credit_amt'] <= 2000), 'credit_amt'] = 0
        df.loc[(df['credit_amt'] > 2000) & (df['credit_amt'] <= 5000), 'credit_amt'] = 1
        df.loc[(df['credit_amt'] > 5000), 'credit_amt'] = 2    
        df.loc[(df['duration'] <= 12), 'duration'] = 0
        df.loc[(df['duration'] > 12) & (df['duration'] <= 24), 'duration'] = 1
        df.loc[(df['duration'] > 24) & (df['duration'] <= 36), 'duration'] = 2
        df.loc[(df['duration'] > 36), 'duration'] = 3
        df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['job'] = df['job'].map({'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}).astype(int)    
    df['telephone'] = df['telephone'].map({'A191': 0, 'A192': 1}).astype(int)
    df['foreign_worker'] = df['foreign_worker'].map({'A201': 1, 'A202': 0}).astype(int)

    return df


def process_law(df):
    df.loc[(df['lsat'] <= 30), 'lsat'] = 0.0
    df.loc[(df['lsat'] > 30), 'lsat'] = 1.0
    df.loc[(df['decile3'] <= 5), 'decile3'] = 0.0
    df.loc[(df['decile3'] > 5), 'decile3'] = 1.0
    df['decile1b'] = pd.cut(df['decile1b'], bins=3, labels=range(3), include_lowest=True).astype(float)
    df.loc[(df['ugpa'] <= 3), 'ugpa'] = 0.0
    df.loc[(df['ugpa'] > 3), 'ugpa'] = 1.0
    df['zfygpa'] = pd.cut(df['zfygpa'], bins=2, labels=range(2), include_lowest=True).astype(float)
    df.loc[(df['zgpa'] <= -1), 'zgpa'] = 0.0
    df.loc[(df['zgpa'] > -1) & (df['zgpa'] <= 1), 'zgpa'] = 1.0
    df.loc[(df['zgpa'] > 1), 'zgpa'] = 2.0
    return df

def process_adult(df):
    # replace missing values (?) to nan and then drop the columns
    df['country'] = df['country'].replace('?',np.nan)
    df['workclass'] = df['workclass'].replace('?',np.nan)
    df['occupation'] = df['occupation'].replace('?',np.nan)
    # dropping the NaN rows now
    df.dropna(how='any',inplace=True)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['workclass'] = df['workclass'].map({'Never-worked': 0, 'Without-pay': 1, 'State-gov': 2, 'Local-gov': 3, 'Federal-gov': 4, 'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'Private': 7}).astype(int)
    df['education'] = df['education'].map({'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad':8, 'Some-college': 9, 'Bachelors': 10, 'Prof-school': 11, 'Assoc-acdm': 12, 'Assoc-voc': 13, 'Masters': 14, 'Doctorate': 15}).astype(int)
    df.loc[(df['education'] <= 4), 'education'] = 0
    df.loc[(df['education'] > 4) & (df['education'] <= 8), 'education'] = 1
    df.loc[(df['education'] > 8) & (df['education'] <= 13), 'education'] = 2
    df.loc[(df['education'] > 13), 'education'] = 3
    df['marital'] = df['marital'].map({'Married-civ-spouse': 2, 'Divorced': 1, 'Never-married': 0, 'Separated': 1, 'Widowed': 1, 'Married-spouse-absent': 2, 'Married-AF-spouse': 2}).astype(int)
    df['relationship'] = df['relationship'].map({'Wife': 1 , 'Own-child': 0 , 'Husband': 1, 'Not-in-family': 0, 'Other-relative': 0, 'Unmarried': 0}).astype(int)
    df['race'] = df['race'].map({'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0, 'Black': 0}).astype(int)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype(int)
    # process hours
    df.loc[(df['hours'] <= 40), 'hours'] = 0
    df.loc[(df['hours'] > 40), 'hours'] = 1
    # process nationality
#     df.loc[(df['country'] != 'United-States'), 'country'] = 0
#     df.loc[(df['country'] == 'United-States'), 'country'] = 1
    df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'country', 'capgain', 'caploss'])
#     df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'capgain', 'caploss'])
    df = df.reset_index(drop=True)
    return df


def process_hmda(df):
    df = df[(df.action_taken==1) | (df.action_taken==3)]
    w_idx = df[(df['applicant_race-1']==5)
               &(pd.isna(df['applicant_race-2']))
               &(df['applicant_ethnicity-1']==2)].index
    b_idx = df[(df['applicant_race-1']==3)& (pd.isna(df['applicant_race-2']))].index
    df['race'] = -1
    df['race'].loc[w_idx] = 1
    df['race'].loc[b_idx] = 0
    df = df[df['race']>=0]
    df = df[df['debt_to_income_ratio']!='Exempt']
    df['gender'] = -1
    df['gender'][df['applicant_sex']==1] = 1
    df['gender'][df['applicant_sex']==2] = 0
    df = df[df['gender']>=0]

    df = df[['action_taken', 'income', 'race', 'gender', 'loan_type', 'applicant_age',
             'debt_to_income_ratio', 'loan_to_value_ratio', 'lien_status']]

    df['income'].fillna(71, inplace=True)
    df['loan_to_value_ratio'].fillna(93, inplace=True)
    df['debt_to_income_ratio'].fillna(41, inplace=True)

    df['applicant_age'] = df['applicant_age'].map({'25-34': 0, '35-44': 0, '<25': 0, '8888': -1,
                                                   '45-54': 1, '55-64': 1, '65-74': 1, '>74': 1})
    df.dropna(inplace=True)
    df['applicant_age'] = df['applicant_age'].astype(int)
    df = df[df['applicant_age']>=0]

    df['loan_to_value_ratio']= pd.to_numeric(df['loan_to_value_ratio'], errors= 'coerce')
    bins = np.array([40, 60, 79, 81, 90, 100])
    df['LV'] = np.ones_like(df['loan_to_value_ratio'])
    df['LV'][df['loan_to_value_ratio']<90] = 0
#     bins = np.array([0, 90, 100])
#     df['LV'] = np.digitize(df['loan_to_value_ratio'], bins)

    df.loc[df['debt_to_income_ratio']=='<20%', 'debt_to_income_ratio'] = 15
    df.loc[df['debt_to_income_ratio']=='20%-<30%', 'debt_to_income_ratio'] = 25
    df.loc[df['debt_to_income_ratio']=='30%-<36%', 'debt_to_income_ratio'] = 33
    df.loc[df['debt_to_income_ratio']=='50%-60%', 'debt_to_income_ratio'] = 55
    df.loc[df['debt_to_income_ratio']=='>60%', 'debt_to_income_ratio'] = 65
    df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'])
    bins = np.array([0, 20, 30, 36, 40, 45, 50, 60])
    bins = np.array([0, 30, 60, 90])
    df['DI'] = np.digitize(df['debt_to_income_ratio'], bins)

    bins = np.array([32, 53, 107, 374])
#     df['income_brackets'] = np.digitize(df['income'], bins)
    df['income_brackets'] = np.ones_like(df['income'])
    df['income_brackets'][df['income']<100] = 0

    df = df.reset_index(drop=True)
    df['action_taken'][df['action_taken']==3] = 0
    
    return df


def process_compas(df):
    df['age_cat'] = df['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}).astype(int)    
    df['score_text'] = df['score_text'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)    
    df['race'] = df['race'].map({'Other': 0, 'African-American': 0, 'Hispanic': 0, 'Native American': 0, 'Asian': 0, 'Caucasian': 1}).astype(int)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(int)    
    
    df.loc[(df['priors_count'] <= 5), 'priors_count'] = 0
    df.loc[(df['priors_count'] > 5) & (df['priors_count'] <= 15), 'priors_count'] = 1
    df.loc[(df['priors_count'] > 15), 'priors_count'] = 2
    
    df.loc[(df['juv_fel_count'] == 0), 'juv_fel_count'] = 0
    df.loc[(df['juv_fel_count'] == 1), 'juv_fel_count'] = 1
    df.loc[(df['juv_fel_count'] > 1), 'juv_fel_count'] = 2
    
    df.loc[(df['juv_misd_count'] == 0), 'juv_misd_count'] = 0
    df.loc[(df['juv_misd_count'] == 1), 'juv_misd_count'] = 1
    df.loc[(df['juv_misd_count'] > 1), 'juv_misd_count'] = 2
    
    df.loc[(df['juv_other_count'] == 0), 'juv_other_count'] = 0
    df.loc[(df['juv_other_count'] == 1), 'juv_other_count'] = 1
    df.loc[(df['juv_other_count'] > 1), 'juv_other_count'] = 2
    return df


def load_german(preprocess=True):
    cols = ['status', 'duration', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employment',\
            'install_rate', 'personal_status', 'debtors', 'residence', 'property', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'credit']
    df = pd.read_table('german.data', names=cols, sep=" ", index_col=False)
    df['credit'] = df['credit'].replace(2, 0) #1 = Good, 2= Bad credit risk
    y = df['credit']
    df = process_german(df, preprocess)
    if preprocess:
        df = df.drop(columns=['purpose', 'personal_status', 'housing', 'credit'])
    else:
        df = df.drop(columns=['personal_status', 'credit'])
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_adult(sample=False):
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital', 'occupation',\
            'relationship', 'race', 'gender', 'capgain', 'caploss', 'hours', 'country', 'income']
    if sample:
        df_train = pd.read_csv('adult-sample-train-10pc', names=cols, sep=",")
        df_test = pd.read_csv('adult-sample-test-10pc', names=cols, sep=",")
    else:
        df_train = pd.read_csv('adult.data', names=cols, sep=",")
        df_test = pd.read_csv('adult.test', names=cols, sep=",")

    df_train = process_adult(df_train)
    df_test = process_adult(df_test)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    X_train = df_train.drop(columns='income')
    y_train = df_train['income']

    X_test = df_test.drop(columns='income')
    y_test = df_test['income']
    return X_train, X_test, y_train, y_test


def load_compas():
    df = pd.read_csv('compas-scores-two-years.csv')
    df = df[['event', 'is_violent_recid', 'is_recid', 'priors_count', 'juv_other_count',\
             'juv_misd_count', 'juv_fel_count', 'race', 'age_cat', 'sex','score_text']]
    df = process_compas(df)

    y = df['is_recid']
    # y = df['is_violent_recid']
    df = df.drop(columns=['is_recid', 'is_violent_recid'])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_traffic():
    df = pd.read_csv('traffic_violations_cleaned.csv')
    y = df['search_outcome']
    df = df.drop(columns=['search_outcome'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_sqf():
    df_train = pd.read_csv('sqf_train.csv')
    y_train = df_train['frisked']
    df_train['inout'] = df_train['inout_I']
    df_train['gender'] = df_train['sex_M']
    
    X_train = df_train.drop(columns=['frisked', 'inout_I', 'inout_O', 'sex_M', 'sex_F'])
    proxy = y_train + X_train.gender*2 - np.random.binomial(n=1, p=0.2, size=len(y_train))
    X_train['proxy'] = np.where(proxy>=1.5, 1, 0)
    
    df_test = pd.read_csv('sqf_test.csv')
    y_test = df_test['frisked']
    df_test['inout'] = df_test['inout_I']
    df_test['gender'] = df_test['sex_M']
    X_test = df_test.drop(columns=['frisked', 'inout_I', 'inout_O', 'sex_M', 'sex_F'])
    proxy = y_test + X_test.gender*2 - np.random.binomial(n=1, p=0.2, size=len(y_test))
    X_test['proxy'] = np.where(proxy>=1.5, 1, 0)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_law():
    df = pd.read_csv('law-school-dataset/law_dataset.csv')
    y = df['pass_bar']
    df = df.drop(columns=['pass_bar'])
    df = process_law(df)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_hmda_ca():
    data = pd.read_csv('races_White-Black or African American_loan_purposes_1_year_2019.csv')
    state = 'CA'
    df = data[data['state_code']==state].reset_index(drop=True)
    df = process_hmda(df)
    y = df['action_taken']
    df = df.drop(columns=['action_taken', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    del data
    return X_train, X_test, y_train, y_test


def load_hmda(subarea=None):
#     data = pd.read_csv('races_White-Black or African American_loan_purposes_1_year_2019.csv')
#     df = process_hmda(data)
#     y = df['action_taken']
#     df = df.drop(columns=['action_taken', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio'])
#     X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
#     X_train = copy.deepcopy(X_train).reset_index(drop=True)
#     X_test = copy.deepcopy(X_test).reset_index(drop=True)
#     y_train = y_train.reset_index(drop=True)
#     y_test = y_test.reset_index(drop=True)
    if subarea is None:
        X_train = pd.read_csv('hmda_X_train.csv', index_col=False)
        X_test = pd.read_csv('hmda_X_test.csv', index_col=False)
        y_train = pd.read_csv('hmda_y_train.csv', index_col=False).action_taken
        y_test = pd.read_csv('hmda_y_test.csv', index_col=False).action_taken
    else:
        X_train = pd.read_csv(f'hmda_{subarea}_X_train.csv', index_col=False)
        X_test = pd.read_csv(f'hmda_{subarea}_X_test.csv', index_col=False)
        y_train = pd.read_csv(f'hmda_{subarea}_y_train.csv', index_col=False).action_taken
        y_test = pd.read_csv(f'hmda_{subarea}_y_test.csv', index_col=False).action_taken
    return X_train, X_test, y_train, y_test


def load_hmda_la():
    data = pd.read_csv('hmda-la-19.csv')
    state = 'LA'
    df = data[data['state_code']==state].reset_index(drop=True)
    df = process_hmda(df)
    y = df['action_taken']
    df = df.drop(columns=['action_taken', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    del data
    return X_train, X_test, y_train, y_test


def load_syn(n_train=10**5, n_test=2*10**4):
    np.random.seed(0)
    data = generate_orig_data_point(n_train + n_test)
    data_train = data[:n_train]
    data_test = data[n_test:]

    X_train = data_train[:, :5]
    y_train = data_train[:, 5]
    X_test = data_test[:, :5]
    y_test = data_test[:, 5]

    X_train_orig = pd.DataFrame({'S': X_train[:, 0], 'X1': X_train[:, 1], 'X2': X_train[:, 2],
                                 'X3': X_train[:, 3], 'X4': X_train[:, 4]})
    X_test_orig = pd.DataFrame({'S': X_test[:, 0], 'X1': X_test[:, 1], 'X2': X_test[:, 2],
                                'X3': X_test[:, 3], 'X4': X_test[:, 4]})

    return X_train_orig, X_test_orig, pd.Series(y_train), pd.Series(y_test)


def load(dataset, preprocess=True, row_num=10000, attr_num=30, sample=False, subarea=None):
    if dataset == 'compas':
        return load_compas()
    elif dataset == 'adult':
        return load_adult(sample=sample)
    elif dataset == 'german':
        return load_german(preprocess)
    elif dataset == 'traffic':
        return load_traffic()
    elif dataset == 'sqf':
        return load_sqf()
    elif dataset == 'law':
        return load_law()
    elif dataset == 'hmda':
        return load_hmda(subarea)
    elif dataset == 'random':
        return generate_random_dataset(row_num, attr_num)
    elif dataset == 'syn':
        return load_syn()
    else:
        raise NotImplementedError
        
        
def generate_random_dataset(row_num, attr_num):
    cols_ls = list()
    for attr_idx in range(attr_num):
        col = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
        cols_ls.append(col)
    X_mat = np.concatenate(cols_ls, axis=1)
    noise = np.random.binomial(n=2, p=0.03, size=(row_num, 1))
    random_coef = np.random.random(attr_num)
    
    X = pd.DataFrame(X_mat, columns=[f'A{attr_idx}' for attr_idx in range(attr_num)])
    y = pd.Series(np.where((np.dot(X_mat, random_coef.reshape(-1, 1))+noise)>attr_num*0.25, 1, 0).ravel(), name='foo')
    X['AA'] = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def generate_orig_data_point(N):
    S = np.random.binomial(1, 0.5, N).reshape(-1, 1)
    X1 = np.where((S + 0.6 * np.random.uniform(0, 1, N).reshape(-1, 1)) > 0.5, 1, 0)
    X3 = np.around(2 * np.random.uniform(0, 1, N).reshape(-1, 1))

    alpha = 1
    beta = 0.2
    X2 = np.where((alpha * X1 + beta * X3 + 0.1 * np.random.uniform(0, 1, N).reshape(-1, 1)) > 0.5, 1, 0)

    Y = np.where((0.3 * X3 + 0.2 * np.random.uniform(-1, 1, N).reshape(-1, 1)) > 0.3, 1, 0)

    X4 = np.where((0.7 * Y + 0.3 * np.random.uniform(-1, 1, N).reshape(-1, 1)) > 0.5, 1, 0)

    return np.concatenate([S, X1, X2, X3, X4, Y], axis=1)