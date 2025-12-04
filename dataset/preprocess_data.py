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

    df = df[(df["year"] >= 1990) & (df["year"] <= 2023)]

    co2 = defaultdict(float)
    for _, row in df.iterrows():
        if row["country"] in relevant_countries:
            co2[row["country"] + " " + str(row["year"])] = row["co2"]

    deltas = defaultdict(float)
    for _, row in df.iterrows():
        if row["country"] in relevant_countries:
            for future_year in range(row["year"], min(2024, row["year"] + 11)):
                delta = co2[row["country"] + " " + str(future_year)] - co2[row["country"] + " " + str(row["year"])]
                deltas[row["country"] + " " + str(future_year) + " " + str(row["year"])] = delta

    prepared_df = pd.DataFrame(list(deltas.items()), columns=['country time range', 'delta co2'])
    return prepared_df


def prepare_policy_data():
    df = pd.read_csv('oecd-policies.csv')
    df = df[['Reference area', 'Climate actions and policies', 'TIME_PERIOD', 'OBS_VALUE']].dropna()
    df = df.loc[df.groupby(['Reference area', 'Climate actions and policies', 'TIME_PERIOD'])['OBS_VALUE'].idxmax()].dropna()

    countries = df['Reference area'].tolist()
    policies = df['Climate actions and policies'].tolist()
    years = df['TIME_PERIOD'].tolist()
    stringency = df['OBS_VALUE'].tolist()
    
    yearly_data = defaultdict(list)

    for _, row in df.iterrows():
        country = row['Reference area']
        if country in CONSISTENT_COUNTRIES:
            country = CONSISTENT_COUNTRIES[country]
        policy = row['Climate actions and policies']
        year = row['TIME_PERIOD']
        stringency = row['OBS_VALUE']
        
        yearly_data[country + " " + str(year) + " " + str(year)].append((stringency, policy))

        for future_year in range(year + 1, min(2024, year + 11)):
            yearly_data[country + " " + str(future_year) + " " + str(year)].append((stringency, policy))

    for k in yearly_data.keys():
        yearly_data[k] = sorted(yearly_data[k])

    prepared_df = pd.DataFrame(list(yearly_data.items()), columns=['country time range', 'policies'])
    return prepared_df

def gen_list_of_countries():
    df = pd.read_csv('oecd-policies.csv')

    relevant_countries = list(set(df["Reference area"].tolist()))
    for country in relevant_countries:
        if country in CONSISTENT_COUNTRIES:
            country = CONSISTENT_COUNTRIES[country]

    return relevant_countries

co2_df = prepare_co2_data(gen_list_of_countries())
policy_df = prepare_policy_data()
df_combined = pd.merge(co2_df, policy_df, on='country time range', how='outer')
df_combined.to_csv('data.csv', index=False)
