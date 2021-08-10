import pandas as pd


def load_data():

    df = pd.read_csv('./assets/BankChurners.csv')
    df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
            'CLIENTNUM'],
        axis=1,
        inplace=True)

    return df

def balance_labels(df, random_state=42):
        attrited = df.loc[df['Attrition_Flag'] == 'Attrited Customer']
        existing = df.loc[df['Attrition_Flag'] == 'Existing Customer']

        balanced_df = pd.concat([attrited.reset_index(drop=True),
                       existing.sample(n=len(attrited), replace=False, random_state=random_state).reset_index(drop=True)])

        return balanced_df


def get_data_target(df):
    target = df['Attrition_Flag']
    data = df.drop(['Attrition_Flag'], axis=1)
    data = pd.get_dummies(data, drop_first=False)

    return data, target
