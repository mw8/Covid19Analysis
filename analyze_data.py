import numpy as np
import datetime as dt

def load_county_pop(county, state):
    import csv
    with open('county_pop.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            if row[0] == county + " County":
                return int(row[2])

def plot_county_list(county_list):
    csv_data = np.loadtxt('covid-19-data/us-counties.csv', delimiter=',', skiprows=1,
                          dtype={'names': ('date', 'county', 'state', 'fips', 'cases', 'deaths'),
                                 'formats': ('S10', 'S32', 'S32', 'S5', 'i4', 'i4')})
    for (county, state) in county_list:
        print((county, state))
        print(load_county_pop(county, state))

if __name__ == "__main__":
    plot_county_list([("Santa Clara", "California"), ("Los Angeles", "California"), ("Broward", "Florida")])
