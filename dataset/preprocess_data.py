import pandas as pd
from collections import defaultdict

CONSISTENT_COUNTRIES = {
    'Slovak Republic': 'Slovakia',
    'Korea': 'South Korea',
    'European Union (27 countries from 01/02/2020)': 'European Union (27)',
    'China (People’s Republic of)': 'China',
    'Türkiye': 'Turkey'
}

def prepare_co2_data(relevant_countries):
    columns = ["country", "year", "co2"]
    df = pd.read_csv("owid-co2.csv", usecols=columns)
    df = df.dropna(subset=["co2"]).copy()

    df = df.sort_values(["country", "year"])

    df = df[(df["year"] >= 1989) & (df["year"] <= 2023)]

    co2 = defaultdict(float)
    for _, row in df.iterrows():
        if row["country"] in relevant_countries:
            co2[row["country"] + "_" + str(row["year"])] = row["co2"]

    delta_co2 = defaultdict(float)
    for _, row in df.iterrows():
        if row["country"] in relevant_countries and row["year"] >= 1990:
            country_time = row["country"] + "_" + str(row["year"])
            country_time2 = row["country"] + "_" + str(row["year"] - 1)
            delta_co2[country_time] = co2[country_time] - co2[country_time2]

    prepared_df = pd.DataFrame(list(delta_co2.items()), columns=['country time', 'delta co2'])
    return prepared_df


def prepare_policy_data():
    df = pd.read_csv('oecd-policies.csv')
    df = df[['Reference area', 'Climate actions and policies', 'TIME_PERIOD', 'OBS_VALUE']].dropna()
    df = df.loc[df.groupby(['Reference area', 'Climate actions and policies', 'TIME_PERIOD'])['OBS_VALUE'].idxmax()].dropna()

    countries = df['Reference area'].tolist()
    policies = df['Climate actions and policies'].tolist()
    years = df['TIME_PERIOD'].tolist()
    stringency = df['OBS_VALUE'].tolist()

    yearly_data = defaultdict(lambda: [[] for i in range(11)])

    for _, row in df.iterrows():
        country = row['Reference area']
        if country in CONSISTENT_COUNTRIES:
            country = CONSISTENT_COUNTRIES[country]

        policy = row['Climate actions and policies']
        year = row['TIME_PERIOD']
        stringency = row['OBS_VALUE']

        yearly_data[country + "_" + str(year)][0].append((stringency, policy))

        for future_year in range(year + 1, min(2024, year + 11)):
            yearly_data[country + "_" + str(future_year)][future_year - year].append((stringency, policy))

    columns = ['country time', 'policies']

    prepared_df = pd.DataFrame(list(yearly_data.items()), columns=columns)
    return prepared_df

def gen_list_of_countries():
    df = pd.read_csv('oecd-policies.csv')
    df = df[['Reference area', 'Climate actions and policies', 'TIME_PERIOD', 'OBS_VALUE']].dropna()
    df = df.loc[df.groupby(['Reference area', 'Climate actions and policies', 'TIME_PERIOD'])['OBS_VALUE'].idxmax()].dropna()

    relevant_countries = list(set(df["Reference area"].tolist()))
    for i in range(len(relevant_countries)):
        if relevant_countries[i] in CONSISTENT_COUNTRIES:
            relevant_countries[i] = CONSISTENT_COUNTRIES[relevant_countries[i]]

    return relevant_countries

co2_df = prepare_co2_data(gen_list_of_countries())
policy_df = prepare_policy_data()
df_combined = pd.merge(co2_df, policy_df, on='country time', how='outer')
df_combined.to_csv('data.csv', index=False)
