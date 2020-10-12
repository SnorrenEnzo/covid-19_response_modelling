import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import urllib.request
import shutil
import os, sys

dataloc = './Data/'
plotloc = './Plots/'
plotloc_government_response = './Gov_response_plots/'

#source: https://www.cbs.nl/nl-nl/visualisaties/bevolkingsteller
n_inhabitants_NL = 17455552
per_million_factor = 1e6/n_inhabitants_NL

#number of days after which an infected person becomes infectuous themselves
serial_interval = 5

#number of days between infection and recovery
time_till_recovery = 14

def downloadSave(url, file_name, check_file_exists = False):
	"""
	Return True when a new file is downloaded
	"""
	if not os.path.exists(file_name):
		print('Downloading file...')
		with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)

		return True

def load_prevalence_R0_data():
	"""
	Load data and return as estimated number of infectious people per million
	"""
	url_prevalence = 'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json'
	url_R0 = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

	fname_prevalence = f'{dataloc}prevalence.json'
	fname_R0 = f'{dataloc}R0.json'

	downloadSave(url_prevalence, fname_prevalence, check_file_exists = True)
	downloadSave(url_R0, fname_R0, check_file_exists = True)

	df_prevalence = pd.read_json(fname_prevalence)
	df_R0 = pd.read_json(fname_R0)

	df_prevalence.set_index('Date', inplace = True)
	df_R0.set_index('Date', inplace = True)

	#change prevalence to per million
	df_prevalence['prev_avg'] *= per_million_factor
	df_prevalence['prev_low'] *= per_million_factor
	df_prevalence['prev_up'] *= per_million_factor

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

def load_government_response_data():
	url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
	fname = f'{dataloc}OxCGRT_latest.csv'

	downloadSave(url, fname, check_file_exists = True)


	df_response = pd.read_csv(fname)

	#filter on the netherlands
	df_response = df_response.loc[df_response['CountryCode'] == 'NLD']

	#select desired columns
	df_response = df_response[['Date', 'StringencyIndex']]

	#read dates
	df_response['Date'] = pd.to_datetime(df_response['Date'], format = '%Y%m%d')

	df_response.set_index('Date', inplace = True)

	return df_response

def load_mobility_data():
	"""
	Load Apple and Google mobility data. Downloadable from:

	Apple: https://covid19.apple.com/mobility
	Google: https://www.google.com/covid19/mobility/

	Apple data:
	- driving
	- transit
	- walking

	Google data:
	- Retail and recreation
	- Grocery and pharmacy
	- Parks
	- Transit stations
	- Workplaces
	- Residential
	"""

	#first load google data, this is the easiest
	google_mobility_fname = f'{dataloc}google_2020_NL_Region_Mobility_Report.csv'

	google_col_rename = {
		'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
		'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
		'parks_percent_change_from_baseline': 'parks',
		'transit_stations_percent_change_from_baseline': 'transit_stations',
		'workplaces_percent_change_from_baseline': 'workplaces',
		'residential_percent_change_from_baseline': 'residential',
		'date': 'date',
		'sub_region_1': 'region'
	}

	df_google_mob = pd.read_csv(google_mobility_fname, usecols = list(google_col_rename.keys()))
	df_google_mob = df_google_mob.rename(columns = google_col_rename)

	#only select country wide data
	country_wide_mask = df_google_mob['region'].isnull()
	df_google_mob = df_google_mob.loc[country_wide_mask]
	del df_google_mob['region']

	#convert date to datetime and set as index
	df_google_mob['date'] = pd.to_datetime(df_google_mob['date'], format = '%Y-%m-%d')
	df_google_mob.set_index('date', inplace = True)


	return df_google_mob

def load_daily_covid():
	url_daily_covid = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'
	fname_daily_covid = f'{dataloc}positieve_tests.csv'

	newdownloaded = downloadSave(url_daily_covid, fname_daily_covid, check_file_exists = True)

	#load only a few columns
	df_daily_covid = pd.read_csv(fname_daily_covid, usecols = ['Date_of_publication', 'Total_reported', 'Hospital_admission', 'Deceased'], sep = ';')

	#rename columns
	df_daily_covid = df_daily_covid.rename(columns = {'Date_of_publication': 'Date'})

	#convert to date
	df_daily_covid['Date'] = pd.to_datetime(df_daily_covid['Date'], format = '%Y-%m-%d %H:%M:%S')

	#aggregate over the dates
	df_daily_covid = df_daily_covid.groupby(['Date']).sum()

	return df_daily_covid

def average_kernel(size = 7):
	return np.ones(size)/size

def get_mean_mu(df_prevalence, df_deaths):
	#slice the dD/dt between dates
	mask = (df_deaths.index > '2020-02-27') & (df_deaths.index <= '2020-08-10')
	df_deaths = df_deaths.loc[mask]

	#do the same for the prevalence I
	mask = (df_prevalence.index > '2020-02-27') & (df_prevalence.index <= '2020-08-10')
	df_prevalence = df_prevalence.loc[mask]

	df_mu = df_deaths[['Amount']].merge(df_prevalence[['prev_avg']], right_index = True, left_index = True)

	#now calculate the mortality rate per unit time
	df_mu['mu'] = df_mu['Amount']/df_mu['prev_avg'] #day^-1

	fig, ax = plt.subplots()

	ax.plot(df_mu.index, df_mu['mu'])

	ax.set_ylabel(r'Mortality rate per unit time ($\mu$) [day$^-1$]')
	ax.set_title('Change in COVID-19 mortality rate per unit time ')

	ax.grid(linestyle = ':')

	plt.savefig('mu_change_NL.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

	#determine average between april and june
	mask = (df_mu.index > '2020-04-01') & (df_mu.index <= '2020-06-01')
	mu_mean = np.mean(df_mu.loc[mask]["mu"])
	print(f'Mean mu: {mu_mean}')

	return mu_mean

def exponential_model(nE_0, R0, t, tau):
	return nE_0 * R0**(t/tau)

def linear_model(x, a, b):
	return a*x + b

def fit_model(model, xdata, ydata, p0 = None, sigma = None):
	#returns the best values for the parameters of the model in the array popt
	#the array pcov contains the estimated covariance of popt
	#p0 are the values for the variables it will start to look
	popt, pcov = curve_fit(model, xdata, ydata, p0 = p0, sigma = sigma)
	#Then the standard deviation is given by:
	perr = np.sqrt(np.diag(pcov))
	#get the residuals:
	residuals = ydata- model(xdata, *popt)
	#to get R^2:
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((ydata-np.mean(ydata))**2)
	r_squared = 1 - (ss_res / ss_tot)

	return popt, perr, r_squared

def plot_prevalence_R():
	df_prevalence, df_R0 = load_prevalence_R0_data()

	print(np.nanmin(df_prevalence['prev_avg']))

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ln1 = ax1.plot(df_prevalence.index, df_prevalence['prev_avg'], label = 'Prevalence')
	ax1.fill_between(df_prevalence.index, df_prevalence['prev_low'], df_prevalence['prev_up'], alpha = 0.4)

	ln2 = ax2.plot(df_R0.index, df_R0['Rt_avg'], color = 'maroon', label = r'$R_0$')
	ax2.fill_between(df_R0.index, df_R0['Rt_low'], df_R0['Rt_up'], alpha = 0.4, color = 'maroon')

	ax1.set_ylim(0)
	ax2.set_ylim(0, 2.6)

	lns = ln1 + ln2
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax1.grid(linestyle = ':', axis = 'x')
	ax2.grid(linestyle = ':', axis = 'y')

	ax1.xaxis.set_tick_params(rotation = 45)

	# fig.autofmt_xdate()

	ax1.set_ylabel('Prevalence (estimated active cases per million)')
	ax2.set_ylabel(r'Basic reproductive number $R_0$')

	ax1.set_title('COVID-19 statistics of the Netherlands')

	plt.savefig(f'{plotloc}coronadashboard_prevalence_R.png', dpi = 200, bbox_inches = 'tight')

def plot_mobility():
	df_google_mob = load_mobility_data()

	#smooth the data
	relevant_keys = [
	'retail_recreation',
	'grocery_pharmacy',
	'parks',
	'transit_stations',
	'workplaces',
	'residential'
	]

	for key in relevant_keys:
		df_google_mob[key + '_smooth'] = np.convolve(df_google_mob[key], average_kernel(size = 7), mode = 'same')

	fig, ax = plt.subplots()

	ax.plot(df_google_mob.index, df_google_mob['retail_recreation_smooth'], label = 'Retail & recreation')
	ax.plot(df_google_mob.index, df_google_mob['grocery_pharmacy_smooth'], label = 'Grocery & pharmacy')
	ax.plot(df_google_mob.index, df_google_mob['parks_smooth'], label = 'Parks')
	ax.plot(df_google_mob.index, df_google_mob['transit_stations_smooth'], label = 'Transit stations')
	ax.plot(df_google_mob.index, df_google_mob['workplaces_smooth'], label = 'Workplaces')
	ax.plot(df_google_mob.index, df_google_mob['residential_smooth'], label = 'Residential')

	ax.set_ylim(-100, 100)
	ax.grid(linestyle = ':')

	ax.set_ylabel('Mobility change relative to baseline [%]')

	ax.set_title('Mobility change in the Netherlands based on Google data')

	ax.legend(loc = 'best')
	ax.xaxis.set_tick_params(rotation = 45)

	plt.savefig(f'{plotloc}Mobility_change.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def government_response_results_simple():
	"""
	Make a simple model of the effects of different responses of a government to
	the coronavirus pandemic
	"""

	def response_model_1(prevalence, prevalence_threshold, R_from_now, t_from_now, response_delay = 14, upward_R = 1.3, downward_R = 0.8):
		"""
		Model that ensures that a R0 below 1 is enforced when the prevalence rises
		above a threshold, though with a certain delay due to the incubation period
		etc
		"""
		'''
		if day_measures_taken == None and prevalence < prevalence_threshold:
			return upward_R, day_measures_taken
		elif day_measures_taken == None and prevalence >= prevalence_threshold:
			day_measures_taken = current_day
			return upward_R, day_measures_taken
		elif day_measures_taken != None and (day_measures_taken + response_delay > current_day):
			return upward_R, day_measures_taken
		elif day_measures_taken != None and (day_measures_taken + response_delay <= current_day):
			if prevalence < prevalence_threshold:
				day_measures_taken = None
			return downward_R, day_measures_taken
		'''
		#determine location in array where we can change the response R
		change_loc = np.argmax((t_from_now - t_from_now[0]) > response_delay)

		if prevalence > prevalence_threshold and R_from_now[change_loc] >= 1:
			R_from_now[change_loc:] = downward_R
		if prevalence < prevalence_threshold and R_from_now[change_loc] < 1:
			R_from_now[change_loc:] = upward_R

		return R_from_now


	df_prevalence, df_R0 = load_prevalence_R0_data()

	response_delay = 14

	### tweakables
	starting_prev = 200 #per million
	prevalence_threshold = 2500 #per million

	upward_R = 1.2
	downward_R = 0.7

	#time range in days
	timestep_size = 1
	t_range = np.arange(0, 280, timestep_size)

	#store prevalence and R0
	prev_array = np.zeros(len(t_range) + 1)
	response_R_array = np.zeros(len(t_range) + 1) + upward_R

	prev_array[0] = starting_prev

	day_measures_taken = None
	for i, t in enumerate(t_range):
		#calculate response, basically changing the R in the future
		response_R_array[i:] = response_model_1(prev_array[i], prevalence_threshold, response_R_array[i:], t_range[i:], response_delay = response_delay, upward_R = upward_R, downward_R = downward_R)

		#number of exposed persons
		prev_array[i+1] = exponential_model(prev_array[i], response_R_array[i], timestep_size, serial_interval)


	#determine duration of light (R > 1) and heavy (R < 1) lockdown
	heavy_start = np.argmax(response_R_array < 1)
	heavy_end = np.argmax(response_R_array[heavy_start:] > 1) + heavy_start
	light_start = heavy_end
	light_end = np.argmax(response_R_array[light_start:] < 1) + light_start

	heavy_duration = heavy_end - heavy_start
	light_duration = light_end - light_start

	light_heavy_ratio = light_duration/heavy_duration

	print(f'Duration of light lockdown: {light_duration} days')
	print(f'Duration of heavy lockdown: {heavy_duration} days')
	print(f'Light/heavy ratio: {light_heavy_ratio:0.03f}')


	fig, ax1 = plt.subplots(figsize = (7, 6))

	ax2 = ax1.twinx()

	ln1 = ax1.plot(t_range, prev_array[:-1], label = 'Prevalence')
	ln2 = ax2.plot(t_range, response_R_array[:-1], label = 'Response R', color = 'maroon')

	ax1.set_xlabel('Days since start of the outbreak')
	ax1.set_ylabel('Number of contagious persons (prevalence) per million')
	ax2.set_ylabel(r'$R$')

	ax1.set_title(f'Simple lockdown scenario')

	#add caption
	caption = f'Scenario with light lockdown R = {upward_R}, heavy lockdown R = {downward_R}. ' \
				+ f'A response is\ntriggered at {prevalence_threshold} cases per million, taking ' \
				+ f'{response_delay} days to produce results.\n' \
				+ f'Resulting ratio of light to heavy lockdown duration: {light_heavy_ratio:0.03f} (higher is better).\n' \
				 + 'Modelling: simple exponential model with number of cases ' \
				 + r'$n(t) = n(0) \cdot R^{\frac{t}{\tau}}$'
	fig.text(0.1, 0.03, caption, ha = 'left')
	#make space
	fig.subplots_adjust(bottom = 0.25)

	lns = ln1 + ln2
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax2.set_ylim(0, np.max(response_R_array) * 2)

	ax1.grid(linestyle = ':')

	plt.savefig(f'{plotloc_government_response}Government_response_outcome_simple_1_{upward_R}_{downward_R}.png', dpi = 200, bbox_inches = 'tight')
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

	plt.savefig(f'{plotloc_government_response}Government_response_outcome_complex.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def stringency_R_correlation():
	df_response = load_government_response_data()
	df_prevalence, df_R0 = load_prevalence_R0_data()

	#merge datasets
	df_reponse_results = df_response.merge(df_R0[['Rt_avg']], right_index = True, left_index = True)

	#select date range
	mask = (df_reponse_results.index > '2020-02-16') & (df_reponse_results.index <= '2020-09-18')
	df_reponse_results = df_reponse_results.loc[mask]

	### fit a linear model
	#select only high stringency to reflect the current day situation
	high_stringency_mask = df_reponse_results['StringencyIndex'] > 30
	popt, perr, r_squared = fit_model(linear_model,
						df_reponse_results.loc[high_stringency_mask]['StringencyIndex'],
						df_reponse_results.loc[high_stringency_mask]['Rt_avg'])



	### plot results
	fig, ax = plt.subplots()

	ax.scatter(df_reponse_results['StringencyIndex'], df_reponse_results['Rt_avg'], color = 'maroon', alpha = 0.6, s = 8, label = 'Measurements')

	#plot the model
	xpoints = np.linspace(np.min(df_reponse_results.loc[high_stringency_mask]['StringencyIndex']), np.max(df_reponse_results.loc[high_stringency_mask]['StringencyIndex']), num = 500)
	ax.plot(xpoints, linear_model(xpoints, *popt), label = r'Fit ($R^2 = $' + f'{r_squared:0.03f})', color = 'black')

	ax.set_xlabel('Oxford Stringency Index')
	ax.set_ylabel(f'$R$')

	ax.grid(linestyle = ':')
	ax.legend(loc = 'best')

	ax.set_title(r'$R$ versus stringency index of Dutch coronavirus reponse')

	plt.savefig(f'{plotloc}Stringency_R_correlation.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def mobility_R_correlation():
	df_prevalence, df_R = load_prevalence_R0_data()
	df_google_mob = load_mobility_data()

	#determine error of R
	df_R['Rt_abs_error'] = ((df_R['Rt_low'] - df_R['Rt_avg']).abs() + (df_R['Rt_up'] - df_R['Rt_avg']).abs())/2

	#merge datasets
	df_mob_R = df_google_mob.merge(df_R[['Rt_avg', 'Rt_abs_error']], right_index = True, left_index = True)

	#select date range
	mask = (df_mob_R.index > '2020-04-01') & (df_mob_R.index <= '2020-09-18')
	df_mob_R = df_mob_R.loc[mask]


	### plot results
	fig, axs = plt.subplots(ncols = 3, nrows = 2, sharex = 'col', sharey = 'row', figsize = (12, 8), gridspec_kw = {'hspace': 0.1, 'wspace': 0})
	axs = axs.flatten()

	key_names = {
	'retail_recreation': 'Retail & recreation',
	'grocery_pharmacy': 'Grocery & pharmacy',
	'parks': 'Parks',
	'transit_stations': 'Transit stations',
	'workplaces': 'Workplaces',
	'residential': 'Residential'
	}


	for i, k in enumerate(key_names.keys()):
		#smooth the data
		df_mob_R[k + '_smooth'] = np.convolve(df_mob_R[k], average_kernel(size = 7), mode = 'same')

		key = k + '_smooth'

		#plot data
		axs[i].scatter(df_mob_R[key], df_mob_R['Rt_avg'], color = 'maroon', alpha = 0.6, s = 8, label = key_names[k])

		### fit a linear model
		popt, perr, r_squared = fit_model(linear_model,
							df_mob_R[key],
							df_mob_R['Rt_avg'],
							sigma = np.array(df_mob_R['Rt_abs_error']))

		#plot the model
		xpoints = np.linspace(np.min(df_mob_R[key]), np.max(df_mob_R[key]), num = 500)
		axs[i].plot(xpoints, linear_model(xpoints, *popt), label = r'Fit ($R^2 = $' + f'{r_squared:0.03f})', color = 'black')

		axs[i].set_title(f'{key_names[k]}')

		# axs[i].set_xlabel('Mobility change relative to baseline [%]')
		# axs[i].set_ylabel(r'$R$')

		axs[i].grid(linestyle = ':')

		if key == 'parks':
			axs[i].set_xlim(right = 100)

		axs[i].legend(loc = 'best')

	#frame for overall x and y labels
	fig.add_subplot(111, frameon = False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.xlabel('Mobility change from baseline [%]')
	plt.ylabel('$R$')



	plt.savefig(f'{plotloc}Mobility_R_correlation.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_daily_results():
	"""
	Plot up to date test results
	"""
	df_daily_covid = load_daily_covid()

	#select second wave of infections
	mask = (df_daily_covid.index > '2020-08-01')
	df_daily_covid = df_daily_covid.loc[mask]

	fig, ax = plt.subplots()

	ax.plot(df_daily_covid.index, df_daily_covid['Total_reported'])

	ax.grid(linestyle = ':')

	ax.set_xlabel('Meldingsdatum (niet datum van afname test)')
	ax.set_ylabel('Aantal positieve tests')

	ax.set_title('Aantal positieve SARS-CoV-2 tests in Nederland')

	# ax.xaxis.set_tick_params(rotation = 45)
	ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax.xaxis.set_major_formatter(myfmt)

	ax.set_ylim(0)

	plt.savefig(f'{plotloc}Positieve_tests_tweede_golf.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

	# print(load_daily_covid)

def estimate_recent_prevalence():
	"""
	Estimate the recent prevalence based on the test positivity ratio
	"""
	df_daily_covid = load_daily_covid()
	df_prevalence, df_R0 = load_prevalence_R0_data()

	df_daily_covid['Total_per_million'] = df_daily_covid['Total_reported'] * per_million_factor

	### correct for the delay between the onset of symptoms and the result of the test
	#for source of incubation period, see the readme
	incubation_period = 8.3 #days
	#test delay determined from anecdotal evidence
	test_delay = 3 #days

	#convert to int for easier data analysis (otherwise we need to interpolate to hours etc)
	total_delay = int(incubation_period + test_delay)

	df_daily_covid['test_delay'] = np.repeat(test_delay, len(df_daily_covid))

	#apply the correction to the testing data
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{total_delay} day')

	#select second wave of infections with the high test rate, but stop at the
	#section where the prevalence flattens (seems unrealistic)
	startdate = '2020-08-01'
	enddate = '2020-09-18'
	df_daily_covid_sel = df_daily_covid.loc[(df_daily_covid.index > startdate) & (df_daily_covid.index < enddate)]
	df_prevalence_sel = df_prevalence.loc[(df_prevalence.index > startdate) & (df_prevalence.index < enddate)]

	#discard days with too low number of positive tests per million for determining the correlation
	test_pos_threshold = 40

	startdate_for_cor = np.argmax(df_daily_covid_sel['Total_per_million'] > test_pos_threshold)
	#datasets for determining correlation
	df_daily_covid_cor = df_daily_covid_sel.loc[df_daily_covid_sel.index > startdate_for_cor]
	df_prevalence_cor = df_prevalence_sel.loc[df_prevalence_sel.index > startdate_for_cor]

	### determine correlation under the assumption that the testing environment
	### so the number of tests, peoples readyness to test etc do not change
	popt, perr, r_squared = fit_model(linear_model,
						df_daily_covid_cor['Total_per_million'],
						df_prevalence_cor['prev_avg'])


	fig, ax = plt.subplots()

	ax.scatter(df_daily_covid_sel['Total_per_million'], df_prevalence_sel['prev_avg'],
					alpha = 0.6, color = 'black', s = 8)

	#plot the model
	xpoints = np.linspace(np.min(df_daily_covid_cor['Total_per_million']), np.max(df_daily_covid_cor['Total_per_million']), num = 500)
	ax.plot(xpoints, linear_model(xpoints, *popt), label = r'Fit ($R^2 = $' + f'{r_squared:0.03f})', color = 'maroon')

	ax.grid(linestyle = ':')

	ax.legend(loc = 'best')

	ax.set_xlabel('Number of cases per million reported per day')
	ax.set_ylabel('Prevalence per million')

	ax.set_title('Correlation between prevalence and number of cases reported per day,\nadjusted for incubation period and test delays (~11 days)')

	plt.savefig(f'{plotloc}Prevalence_vs_positive_tests_correlation.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


	### Now make predictions for the most recent data
	#select the positive test data
	df_daily_covid_pred = df_daily_covid.loc[df_daily_covid.index > enddate]
	#make the predictions
	df_daily_covid_pred['Prev_pred'] = linear_model(df_daily_covid_pred['Total_per_million'], *popt)

	#append the last data point from the real data
	df_prevalence_pred = df_daily_covid_pred[['Prev_pred']]
	df_prevalence_pred = df_prevalence_pred.append(df_prevalence_sel.iloc[-1:][['prev_avg']].rename(columns = {'prev_avg': 'Prev_pred'}))
	#then sort again
	df_prevalence_pred = df_prevalence_pred.sort_index()

	#now plot together with the old data
	fig, ax = plt.subplots()

	#old data
	ax.plot(df_prevalence_sel.index, df_prevalence_sel['prev_avg'], label = 'Measurements', color = 'royalblue')
	#show error bars
	ax.fill_between(df_prevalence_sel.index, df_prevalence_sel['prev_low'], df_prevalence_sel['prev_up'], alpha = 0.4, color = 'royalblue')

	#predictions
	ax.plot(df_prevalence_pred.index, df_prevalence_pred['Prev_pred'], label = 'Prediction', color = 'maroon')

	ax.grid(linestyle = ':')

	ax.set_ylabel('Prevalence per million')

	ax.set_title('COVID-19 estimated active cases in the Netherlands\nwith prediction of recent days')

	ax.legend(loc = 'best')

	# ax.xaxis.set_tick_params(rotation = 45)
	# ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	#set location of date ticks, as the automatic technique does not work
	ax.set_xticks(pd.date_range(np.min(df_prevalence_sel.index), np.max(df_prevalence_pred.index), freq = 'W'))

	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax.xaxis.set_major_formatter(myfmt)

	ax.set_ylim(0)

	plt.savefig(f'{plotloc}Prevalence_second_wave_with_predictions.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

	#make predictions

def main():
	# government_response_results_simple()

	# mobility_R_correlation()

	# estimate_recent_prevalence()

	plot_prevalence_R()

	'''
	df_prevalence, df_R0 = load_prevalence_R0_data()
	df_deaths = load_deathdata()

	### determine mortality rate per unit time mu
	### mu = 1/I dD/dt
	mu_mean = get_mean_mu(df_prevalence, df_deaths) #days^-1

	### set other model parameters
	# period of being not contagious
	a = 1/8.3 #days^-1
	#
	'''


if __name__ == '__main__':
	main()
