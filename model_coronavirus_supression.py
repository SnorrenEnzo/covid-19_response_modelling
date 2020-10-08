import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import urllib.request
import shutil
import os

dataloc = './Data/'

#source: https://www.cbs.nl/nl-nl/visualisaties/bevolkingsteller
n_inhabitants_NL = 17455552
per_million_factor = 1e6/n_inhabitants_NL

#number of days after which an infected person becomes infectuous themselves
serial_interval = 5

#number of days between infection and recovery
time_till_recovery = 14

def downloadSave(url, file_name, check_file_exists = False):
	if not os.path.exists(file_name):
		print('Downloading file...')
		with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)

def load_prevalence_R0_data():
	url_prevalence = 'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json'
	url_R0 = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

	fname_prevalence = f'{dataloc}prevalence.json'
	fname_R0 = f'{dataloc}R0.json'

	downloadSave(url_prevalence, fname_prevalence, check_file_exists = True)
	downloadSave(url_R0, fname_R0, check_file_exists = True)

	df_prevalence = pd.read_json(fname_prevalence)
	df_R0 = pd.read_json(fname_R0)

	df_prevalence.set_index('Date', inplace = True)

	return df_prevalence, df_R0

def load_deathdata():
	df_deaths = pd.read_csv(f'{dataloc}deaths_per_day.csv', sep = ';')
	#add year
	df_deaths['Datum van overlijden'] += ' 2020'
	#replace month with number
	dutchmonths = {
					'feb': 2,
					'mrt': 3,
					'apr': 4,
					'mei': 5,
					'jun': 6,
					'jul': 7,
					'aug': 8,
					'sep': 9,
					'okt': 10,
					'nov': 11,
					'dec': 12
					}
	for monthname, monthidx in zip(dutchmonths.keys(), dutchmonths.values()):
		df_deaths['Datum van overlijden'] = df_deaths['Datum van overlijden'].str.replace(monthname, str(monthidx))
	df_deaths['Datum van overlijden'] = pd.to_datetime(df_deaths['Datum van overlijden'], format = '%d %m %Y')

	#rename columns
	df_deaths = df_deaths.rename(columns = {'Datum van overlijden': 'Date', 't/m afgelopen week': 'Amount'})

	df_deaths.set_index('Date', inplace = True)

	return df_deaths

def exponential_model(nE_0, R0, t, tau):
	return nE_0 * R0**(t/tau)

def plotIRLstats():
	df_prevalence, df_R0 = loaddata()

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ln1 = ax1.plot(df_prevalence['Date'], df_prevalence['prev_avg']*per_million_factor, label = 'Prevalence')
	ax1.fill_between(df_prevalence['Date'], df_prevalence['prev_low']*per_million_factor, df_prevalence['prev_up']*per_million_factor, alpha = 0.4)

	ln2 = ax2.plot(df_R0['Date'], df_R0['Rt_avg'], color = 'maroon', label = r'$R_0$')
	ax2.fill_between(df_R0['Date'], df_R0['Rt_low'], df_R0['Rt_up'], alpha = 0.4, color = 'maroon')

	ax1.set_ylim(0)
	ax2.set_ylim(0, 2.6)

	lns = ln1 + ln2
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax1.grid(linestyle = ':', axis = 'x')
	ax2.grid(linestyle = ':', axis = 'y')

	fig.autofmt_xdate()

	ax1.set_ylabel('Prevalence (estimated active cases per million)')
	ax2.set_ylabel(r'Basic reproductive number $R_0$')

	ax1.set_title('COVID-19 statistics of the Netherlands')

	plt.savefig('coronadashboard_measurements.png', dpi = 200, bbox_inches = 'tight')

def government_response_results_simple():
	df_prevalence, df_R0 = load_prevalence_R0_data()

	peak_prev = 200000

	response_R = 0.9

	#time range in days
	t_range = np.arange(0, 150, 1)

	#number of exposed persons
	nE = exponential_model(peak_prev, response_R, t_range, serial_interval)



	fig, ax = plt.subplots()


	ax.plot(t_range, nE, label = 'Number of contagious persons')

	ax.grid(linestyle = ':')

	plt.savefig('Government_response_outcome_simple.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def government_response_results_SEIRD():
	df_prevalence, df_R0 = load_prevalence_R0_data()

	peak_prev = 200000

	response_R = 0.9

	#number of infections per unit time per person
	infection_rate = 1.5
	# non_contagious_rate = 1/

	#time range in days
	t_range = np.arange(0, 150, 1)

	#number of exposed persons
	nE = exponential_model(peak_prev, response_R, t_range, serial_interval)



	fig, ax = plt.subplots()


	ax.plot(t_range, nE, label = 'Number of contagious persons')

	ax.grid(linestyle = ':')

	plt.savefig('Government_response_outcome_complex.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


def main():
	# plotIRLstats()

	# government_response_results()

	### determine mortality rate per unit time mu
	### mu = 1/I dD/dt
	df_prevalence, df_R0 = load_prevalence_R0_data()

	df_deaths = load_deathdata()

	#slice the dD/dt between dates
	mask = (df_deaths.index > '2020-02-27') & (df_deaths.index <= '2020-08-10')
	df_deaths = df_deaths.loc[mask]

	#do the same for the prevalence I
	mask = (df_prevalence.index> '2020-02-27') & (df_prevalence.index <= '2020-08-10')
	df_prevalence = df_prevalence.loc[mask]

	df_mu = df_deaths[['Amount']].merge(df_prevalence[['prev_avg']], right_index = True, left_index = True)

	#now calculate the mortality rate per unit time
	df_mu['mu'] = df_mu['Amount']/df_mu['prev_avg'] #day^-1

	fig, ax = plt.subplots()

	ax.plot(df_mu.index, df_mu['mu'])

	ax.grid(linestyle = ':')

	plt.savefig('mu.png')


if __name__ == '__main__':
	main()
