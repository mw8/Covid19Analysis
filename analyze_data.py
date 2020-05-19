import numpy as np
import pandas as pd

# load population data for a county
def load_county_pop(county, state):
    import csv
    with open('county_pop.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            if row[0] == county + " County":
                return float(row[2])

# load case and death data for a county
def load_county_cd(csv_data, county, state):
    r = csv_data[np.logical_and(csv_data['county'] == county, csv_data['state'] == state)]
    return np.array(r[['cases', 'deaths']])

# plot cases over time and new cases versus total cases
def plot_county_list(county_list):
    csv_data = pd.read_csv('covid-19-data/us-counties.csv')
    for (county, state) in county_list:
        ccd = load_county_cd(csv_data, county, state)
        pop = load_county_pop(county, state)
        print(ccd)

if __name__ == "__main__":
    plot_county_list([('Santa Clara', 'California'), ('Los Angeles', 'California'), ('Broward', 'Florida')])
