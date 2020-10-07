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

def downloadSave(url, file_name, check_file_exists = False):
	if not os.path.exists(file_name):
		print('Downloading file...')
		with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)

def plotIRLstats():
	url_prevalence = 'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json'
	url_R0 = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

	fname_prevalence = f'{dataloc}prevalence.json'
	fname_R0 = f'{dataloc}R0.json'

	downloadSave(url_prevalence, fname_prevalence, check_file_exists = True)
	downloadSave(url_R0, fname_R0, check_file_exists = True)

	df_prevalence = pd.read_json(fname_prevalence)
	df_R0 = pd.read_json(fname_R0)

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

	ax1.set_ylabel('COVID-19 Prevalence (estimated active cases per million)')
	ax2.set_ylabel(r'Basic reproductive number $R_0$')

	plt.savefig('coronadashboard_measurements.png', dpi = 200, bbox_inches = 'tight')

def main():
	plotIRLstats()

if __name__ == '__main__':
	main()
