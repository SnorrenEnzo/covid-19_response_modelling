import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import urllib.request
from urllib.error import HTTPError
import shutil
import os, sys, glob
from zipfile import ZipFile

import datetime as dt

from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

dataloc = './Data/'
plotloc = './Plots/'
plotloc_government_response = './Gov_response_plots/'
mobplotloc = f'{plotloc}Mobility/'

#source: https://www.cbs.nl/nl-nl/visualisaties/bevolkingsteller
n_inhabitants_NL = 17455552
per_million_factor = 1e6/n_inhabitants_NL

#number of days after which an infected person becomes infectuous themselves
serial_interval = 5

#number of days between infection and recovery
time_till_recovery = 14

betterblue = '#016FB9'
betterorange = '#F17105'
betterblack = '#141414'

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
	raw_fname = f'{dataloc}OxCGRT_latest.csv'

	#filename of the csv data containing only the desired dutch data
	nl_fname = f'{dataloc}OxCGRT_NL.csv'

	#read the data from the Netherlands only, this allows you to easily add extra
	#data which the Oxfort team has not yet added themselves (recent response etc)
	try:
		df_response = pd.read_csv(nl_fname)

		#read dates
		df_response['Date'] = pd.to_datetime(df_response['Date'], format = '%Y-%m-%d')
		df_response.set_index('Date', inplace = True)
	except OSError:
		downloadSave(url, raw_fname, check_file_exists = True)

		df_response = pd.read_csv(raw_fname)

		#filter on the netherlands
		df_response = df_response.loc[df_response['CountryCode'] == 'NLD']

		#select desired columns
		df_response = df_response[['Date', 'StringencyIndex']]

		#read dates
		df_response['Date'] = pd.to_datetime(df_response['Date'], format = '%Y%m%d')
		df_response.set_index('Date', inplace = True)

		df_response.to_csv(nl_fname)

	return df_response

def load_mobility_data(smooth = False, smoothsize = 7, apple_mobility_url_base = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2021HotfixDev18/v3/en-us/applemobilitytrends-'):
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

	### first load google data
	#download zip
	google_mobility_url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
	google_mobility_fname_zip = f'{dataloc}google_2020_All_Region_Mobility_Report.zip'
	google_mobility_fname = f'2020_NL_Region_Mobility_Report.csv'
	downloaded = downloadSave(google_mobility_url, google_mobility_fname_zip, check_file_exists = True)

	if downloaded:
		#unzip
		unziploc = f'{dataloc}unzip/'
		with ZipFile(google_mobility_fname_zip, 'r') as zfile:
			os.mkdir(unziploc)
			zfile.extractall(unziploc)

		#get the desired file
		zipfilename = glob.glob(unziploc + google_mobility_fname)[0]
		print(zipfilename)
		#rename
		google_mobility_fname = dataloc + 'google_' + google_mobility_fname
		os.rename(zipfilename, google_mobility_fname)

		#remove downloaded files
		shutil.rmtree(unziploc)
		#also remove zip file
		# os.remove(google_mobility_fname_zip)
	else:
		google_mobility_fname = dataloc + 'google_' + google_mobility_fname

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


	### now load the apple data
	#the url changes to the most recent date of availability, so we need to scan
	#several urls of the past few days
	possible_apple_data_files = glob.glob(f'{dataloc}applemobilitytrends-*.csv')
	if len(possible_apple_data_files) == 0:

		today = dt.datetime.now().date()
		apple_mob_fname = f'{dataloc}applemobilitytrends-'
		#make the possible urls, starting with the most recent date
		for minday in range(8):
			possible_date = today - dt.timedelta(days = minday)
			apple_mob_possible_url = f'{apple_mobility_url_base}{possible_date}.csv'

			try:
				downloadSave(apple_mob_possible_url, apple_mob_fname + f'{possible_date}.csv', check_file_exists = True)

				print(f'{possible_date} correct!')

				apple_mob_fname += f'{possible_date}.csv'
				break
			except HTTPError:
				print(f'{possible_date} no data found')
	else:
		apple_mob_fname = possible_apple_data_files[0]

	#load any apple mobility file
	# apple_part_fname = f'{dataloc}applemobilitytrends*.csv'
	# apple_mob_fname = glob.glob(apple_part_fname)[0]

	df_apple_mob = pd.read_csv(apple_mob_fname)
	#find the driving, walking & transit rows of the netherlands
	df_apple_mob = df_apple_mob.loc[df_apple_mob['region'] == 'Netherlands']

	#change index to the transportation type
	df_apple_mob.set_index('transportation_type', inplace = True)

	df_apple_mob = df_apple_mob.transpose()

	#remove first rows
	df_apple_mob = df_apple_mob.iloc[6:]

	df_apple_mob.index = pd.to_datetime(df_apple_mob.index, format = '%Y-%m-%d')

	#make the change from 0 instead of 100%
	for col in df_apple_mob.columns:
		df_apple_mob[col] -= 100

	### smooth data if desired
	if smooth:
		for col in df_google_mob.columns:
			df_google_mob[col + '_smooth'] = df_google_mob[col].rolling(smoothsize).mean()

		for col in df_apple_mob.columns:
			df_apple_mob[col + '_smooth'] = df_apple_mob[col].rolling(smoothsize).mean()

	### interpolate the gap in the Apple data
	#resample to once a day
	df_apple_mob = df_apple_mob.resample('1d').mean()

	#then interpolate
	df_apple_mob = df_apple_mob.interpolate(method = 'linear')

	return df_google_mob, df_apple_mob

def load_daily_covid(correct_for_delay = False):
	url_daily_covid = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'
	fname_daily_covid = f'{dataloc}positieve_tests.csv'

	downloadSave(url_daily_covid, fname_daily_covid, check_file_exists = True)

	#load only a few columns
	df_daily_covid = pd.read_csv(fname_daily_covid, usecols = ['Date_of_publication', 'Total_reported', 'Hospital_admission', 'Deceased'], sep = ';')

	#rename columns
	df_daily_covid = df_daily_covid.rename(columns = {'Date_of_publication': 'Date'})

	#convert to date
	df_daily_covid['Date'] = pd.to_datetime(df_daily_covid['Date'], format = '%Y-%m-%d %H:%M:%S')

	#aggregate over the dates
	df_daily_covid = df_daily_covid.groupby(['Date']).sum()

	if correct_for_delay:
		### correct for the delay between the onset of symptoms and the result of the test
		#for source of incubation period, see the readme
		incubation_period = 6 #days, left of average of 8.3 due to extremely skewed distribution
		#test delay determined from anecdotal evidence
		test_delay = 3 #days

		#convert to int for easier data analysis (otherwise we need to interpolate to hours etc)
		total_delay = int(incubation_period + test_delay)

		# df_daily_covid['total_delay'] = np.repeat(total_delay, len(df_daily_covid))

		#apply the correction to the testing data
		df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{total_delay} day')

	return df_daily_covid

def load_number_of_tests(enddate = None):
	"""
	Data source:
	https://www.rivm.nl/documenten/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland

	Compiled into a .csv by hand
	"""

	df_n_tests = pd.read_csv(f'{dataloc}Edit_only/tests_per_week.csv', usecols = ['Week_number', 'Number_of_tests'])

	df_n_tests = df_n_tests.astype({'Week_number': str})

	#get date from week number, assume the central day of the week (thursday)
	df_n_tests['Date'] = pd.to_datetime(df_n_tests['Week_number'] + '-4-2020', format = '%W-%w-%Y')

	df_n_tests.set_index('Date', inplace = True)

	del df_n_tests['Week_number']

	### interpolate to each day
	#first convert units to per day
	df_n_tests['Number_of_tests'] /= 7

	#and ignore last known date (due to incomplete data) if given a final date
	if enddate != None:
		df_n_tests = df_n_tests[:-1]

	#resample to once a day
	df_n_tests = df_n_tests.resample('1d').mean()

	#copy the column for inspection of the interpolation results
	df_n_tests['Number_of_tests_raw'] = df_n_tests['Number_of_tests']

	#then interpolate/extrapolate
	df_n_tests['Number_of_tests'] = df_n_tests['Number_of_tests'].interpolate(method = 'linear')

	#indicate which part was extrapolated
	df_n_tests['Extrapolated'] = 0

	#now if the enddate is given, we also want to extrapolate
	if enddate != None:
		#select on which section we want to extrapolate (up to 3 weeks back)
		start_extrapolation_pred = df_n_tests.index[-1]
		start_extrapolation = start_extrapolation_pred - pd.Timedelta(f'7 day')
		df_extrap_train = df_n_tests.loc[df_n_tests.index > start_extrapolation]
		#get a numerical index
		X_train = df_extrap_train.reset_index().drop('Date', 1).index.astype(float).values
		Y_train = df_extrap_train['Number_of_tests'].values

		## train model
		clf = LinearRegression()
		clf.fit(X_train[:,None], Y_train)

		## now make the proper array
		#set the enddate
		df_n_tests.loc[enddate] = [np.nan, np.nan, 1]
		#add dates in between
		df_n_tests = df_n_tests.resample('1d').mean()

		#extract the rows for extrapolation
		df_extrap_pred = df_n_tests.loc[df_n_tests.index > start_extrapolation]
		selectindex = (df_extrap_pred.index > start_extrapolation_pred)
		#get numerical index
		X_pred = df_extrap_pred.reset_index().drop('Date', 1).index.astype(float).values[selectindex]
		#get predictions
		df_temp = pd.DataFrame(data = clf.predict(X_pred[:,None]), index = df_extrap_pred.loc[df_extrap_pred.index > start_extrapolation_pred].index, columns = ['Number_of_tests'])
		#indicate that these values are extrapolated
		df_temp['Extrapolated'] = 1

		#now update the interpolated array
		df_n_tests.update(df_temp)

		df_n_tests = df_n_tests.astype({'Extrapolated': 'bool'})

	return df_n_tests

def load_sewage_data(smooth = False, windowsize = 3, shiftdates = False):
	sewage_url = 'https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv'
	sewage_fname = f'{dataloc}COVID-19_rioolwaterdata.csv'

	downloadSave(sewage_url, sewage_fname, check_file_exists = True)

	df_sewage = pd.read_csv(sewage_fname, usecols = ['Date_measurement', 'Security_region_name', 'Percentage_in_security_region', 'RNA_flow_per_100000', 'Representative_measurement'])

	#shift RNA units to "* 100 billion"
	df_sewage['RNA_flow_per_100000'] /= 100e9

	#only take "representative measurements" which span 24 hours instead of a single moment
	df_sewage = df_sewage.loc[df_sewage['Representative_measurement']]
	del df_sewage['Representative_measurement']

	#rename columns
	df_sewage = df_sewage.rename(columns = {'Date_measurement': 'Date'})

	#convert to date
	df_sewage['Date'] = pd.to_datetime(df_sewage['Date'], format = '%Y-%m-%d')

	df_sewage.set_index('Date', inplace = True)

	#add column indicating number of measurements
	df_sewage['n_measurements'] = np.ones(len(df_sewage), dtype = int)

	# df_temp = df_sewage.groupby(['Security_region_name']).agg({'RNA_per_ml': 'median', 'n_measurements': 'sum'})

	#aggregate over the dates
	df_sewage = df_sewage.groupby(['Date']).agg({'RNA_flow_per_100000': 'median', 'n_measurements': 'sum'})

	if shiftdates:
		#rough average peak of virus shedding
		peak_shedding_period = 4
		#shift dates to correct for delay between infection and peak virus emission
		df_sewage.index = df_sewage.index - pd.Timedelta(f'{peak_shedding_period} day')

	#smooth data if desired
	if smooth:
		df_sewage['RNA_flow_smooth'] = df_sewage['RNA_flow_per_100000'].rolling(windowsize).mean()

	return df_sewage

def load_IC_data():
	"""
	Load IC data, see https://www.databronnencovid19.nl/Bron?naam=Nationale-Intensive-Care-Evaluatie
	"""
	IC_count_url = 'https://stichting-nice.nl/covid-19/public/intake-count/'
	IC_new_url = 'https://stichting-nice.nl/covid-19/public/new-intake/'

	IC_count_fname = f'{dataloc}IC_count.json'
	IC_new_fname = f'{dataloc}IC_new.json'

	downloadSave(IC_count_url, IC_count_fname, check_file_exists = True)
	downloadSave(IC_new_url, IC_new_fname, check_file_exists = True)

	#the IC_new file has one set of square brackets too many, so lets remove them
	tempfile = 'temp.temp'
	with open(IC_new_fname, 'r') as fin, open(tempfile, 'w') as fout:
		filecontent = fin.read()

		if filecontent.count('[') > 1:
			#remove the appended array
			filecontent = filecontent.split(',[')[0]
			#remove first bracket
			filecontent = filecontent[1:]

		fout.write(filecontent)

		os.rename(tempfile, IC_new_fname)

	df_IC_count = pd.read_json(IC_count_fname)
	df_IC_new = pd.read_json(IC_new_fname)

	df_IC_count = df_IC_count.rename(columns = {'date': 'Date', 'value': 'Amount'})
	df_IC_new = df_IC_new.rename(columns = {'date': 'Date', 'value': 'New'})

	df_IC_count['Date'] = pd.to_datetime(df_IC_count['Date'], format = '%Y-%m-%d')
	df_IC_new['Date'] = pd.to_datetime(df_IC_new['Date'], format = '%Y-%m-%d')

	df_IC_count.set_index('Date', inplace = True)
	df_IC_new.set_index('Date', inplace = True)

	#join dataframes
	df_IC = df_IC_count.join(df_IC_new)

	#determine how many people left the IC (alive or not)
	df_IC['Removed'] = df_IC['Amount'].diff() - df_IC['New']
	#set first entry to zero so that we can convert to ints again
	df_IC['Removed'].iloc[0] = 0
	df_IC['Removed'] = df_IC['Removed'].astype(int)
	#there are a few cases in this field where there are extra people coming in
	#probably due to errors in reporting. Shift these to new
	loc = df_IC['Removed'] > 0
	df_IC['New'].values[loc] += df_IC['Removed'].values[loc]
	df_IC['Removed'].values[loc] = 0

	df_IC['Removed'] = df_IC['Removed'].abs()

	return df_IC

def load_superspreader_events_data():
	"""
	Load data from https://docs.google.com/spreadsheets/d/1c9jwMyT1lw2P0d6SDTno6nHLGMtpheO9xJyGHgdBoco/edit#gid=1812932356

	Description:
	https://kmswinkels.medium.com/covid-19-superspreading-events-database-4c0a7aa2342b
	"""

	fname = f'{dataloc}SARS-CoV-2 Superspreading Events from Around the World - SARS-CoV-2 Superspreading Events.csv'

	load_cols = [
	'Setting1',
	'Indoor / Outdoor',
	'Total Cases',
	'Index Cases',
	'Secondary Cases',
	'Total Pop at Event',
	'Tertiary Cases',
	'Index Date (Day-Month)'
	]

	df_SSE = pd.read_csv(fname, usecols = load_cols)
	df_SSE = df_SSE.rename(columns = {'Index Date (Day-Month)': 'Index Date'})

	df_SSE['Index Date'] = pd.to_datetime(df_SSE['Index Date'])
	df_SSE.set_index('Index Date', inplace = True)

	#filter out all occurances where the 'Total Cases' column is filled with a string
	#instead of a number
	df_SSE = df_SSE.loc[df_SSE['Total Cases'].str.isnumeric()]
	#now convert to int
	df_SSE['Total Cases'] = df_SSE['Total Cases'].astype(int)
	df_SSE['Setting1'] = df_SSE['Setting1'].astype(str)
	df_SSE['Indoor / Outdoor'] = df_SSE['Indoor / Outdoor'].astype(str)

	#filter out unknowns
	# df_SSE.loc[df_SSE['Indoor / Outdoor'] == 'nan']['Indoor / Outdoor'] = 'Unknown'
	# df_SSE.loc[df_SSE['Indoor / Outdoor'] == 'unknown']['Indoor / Outdoor'] = 'Unknown'
	# df_SSE = df_SSE.set_value(df_SSE.index[(df_SSE['Indoor / Outdoor'] == 'nan') | (df_SSE['Indoor / Outdoor'] == 'unknown')], 'Indoor / Outdoor', 'Unknown')

	return df_SSE


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

def three_variables_linear_model(xdata, a, b, c, d):
	return a*xdata[0] + b*xdata[1] + c*xdata[2] + d

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

def dataframes_to_NDarray(df, columns):
	X = np.zeros((len(df), len(columns)))
	for i in range(len(columns)):
		X[:,i] = np.array(df[columns[i]])

	return X


def plot_prevalence_R():
	df_prevalence, df_R0 = load_prevalence_R0_data()
	df_response = load_government_response_data()

	#filter on starting date
	startdate = '2020-02-15'
	df_prevalence = df_prevalence.loc[df_prevalence.index > startdate]
	df_R0 = df_R0.loc[df_R0.index > startdate]
	df_response = df_response.loc[df_response.index > startdate]


	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax3 = ax1.twinx()
	ax3.spines['right'].set_position(('axes', 1.12))

	#plot prevalence
	ln1 = ax1.plot(df_prevalence.index, df_prevalence['prev_avg'], label = 'Prevalence')
	ax1.fill_between(df_prevalence.index, df_prevalence['prev_low'], df_prevalence['prev_up'], alpha = 0.4)

	#plot R
	ln2 = ax2.plot(df_R0.index, df_R0['Rt_avg'], color = 'maroon', label = r'$R$')
	ax2.fill_between(df_R0.index, df_R0['Rt_low'], df_R0['Rt_up'], alpha = 0.4, color = 'maroon')

	#also plot government response
	ln3 = ax3.plot(df_response.index, df_response['StringencyIndex'], label = 'Stringency index', color = betterblack)

	ax1.set_ylim(0)
	ax2.set_ylim(0, 2.6)
	ax3.set_ylim(0)

	lns = ln1 + ln2 + ln3
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax1.grid(linestyle = ':', axis = 'x')
	ax2.grid(linestyle = ':', axis = 'y')

	ax1.xaxis.set_tick_params(rotation = 45)

	# fig.autofmt_xdate()

	ax1.set_ylabel('Prevalence (estimated active cases per million)')
	ax2.set_ylabel(r'Reproductive number $R$')
	ax3.set_ylabel('Oxford Stringency Index')

	ax1.set_title('COVID-19 statistics of the Netherlands')

	plt.savefig(f'{plotloc}coronadashboard_prevalence_R.png', dpi = 200, bbox_inches = 'tight')

def plot_mobility():
	df_google_mob, df_apple_mob = load_mobility_data(smooth = True)

	#smooth the data
	# relevant_keys = [
	# 'retail_recreation',
	# 'grocery_pharmacy',
	# 'parks',
	# 'transit_stations',
	# 'workplaces',
	# 'residential'
	# ]

	### plot google data
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

	ax.legend(loc = 'lower center', ncol = 3, prop = {'size': 9})
	ax.xaxis.set_tick_params(rotation = 45)

	plt.savefig(f'{mobplotloc}Mobility_change_Google.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


	### plot Apple data
	fig, ax = plt.subplots()

	ax.plot(df_apple_mob.index, df_apple_mob['driving_smooth'], label = 'Driving')
	ax.plot(df_apple_mob.index, df_apple_mob['transit_smooth'], label = 'Transit')
	ax.plot(df_apple_mob.index, df_apple_mob['walking_smooth'], label = 'Walking')

	# ax.set_ylim(-100, 100)
	ax.grid(linestyle = ':')

	ax.set_ylabel('Mobility change relative to baseline [%]')

	ax.set_title('Mobility change in the Netherlands based on Apple data')

	ax.legend(loc = 'lower center', ncol = 3, prop = {'size': 9})
	ax.xaxis.set_tick_params(rotation = 45)

	plt.savefig(f'{mobplotloc}Mobility_change_Apple.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_sewage():
	"""
	Plot measurements of SARS-CoV-2 RNA per ml measurements in sewage
	"""
	df_sewage = load_sewage_data(smooth = True, shiftdates = True)

	df_sewage = df_sewage.loc[df_sewage.index > '2020-02-05']

	print(df_sewage.tail())

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ln1 = ax1.plot(df_sewage.index, df_sewage.n_measurements, color = '#0C83CC', label = 'Number of measurements', alpha = 1)

	ax2.scatter(df_sewage.index, df_sewage['RNA_flow_per_100000'], color = 'maroon', label = 'Average RNA abundance', alpha = 0.4, s = 5)
	ln2 = ax2.plot(df_sewage.index, df_sewage['RNA_flow_smooth'], color = 'maroon', label = 'Average RNA abundance smoothed')

	ax1.xaxis.set_tick_params(rotation = 45)

	lns = ln1 + ln2
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax1.set_ylabel('Number of measurements')
	ax2.set_ylabel('Number of RNA fragments per 100 000\ninhabitants' + r' ($\times 100 \cdot 10^9$)')

	ax1.grid(linestyle = ':', axis = 'x')
	ax2.grid(linestyle = ':', axis = 'y')

	ax1.set_title('SARS-CoV-2 RNA abundance in Dutch sewage discharge')

	plt.savefig(f'{plotloc}Sewage_measurements.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_daily_results():
	"""
	Plot up to date test results
	"""
	df_daily_covid = load_daily_covid(correct_for_delay = False)
	df_n_tests = load_number_of_tests(enddate = df_daily_covid.index.values[-1])

	df_response = load_government_response_data()

	### correct for the delay between the onset of symptoms and the result of the test
	#for source of incubation period, see the readme
	incubation_period = 6 #days, left of average of 8.3 due to extremely skewed distribution
	#test delay determined from anecdotal evidence
	time_to_test_delay = 3 #days
	result_delay = 1

	#correct the results to the day of the test
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{result_delay} day')

	#merge datasets
	df_daily_covid = df_daily_covid.merge(df_n_tests[['Number_of_tests', 'Extrapolated']], right_index = True, left_index = True)

	print(df_n_tests.tail())

	#determine test positivity rate
	df_daily_covid['Positivity_ratio'] = df_daily_covid['Total_reported']/df_daily_covid['Number_of_tests']

	#correct the results to the day of infection
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{int(incubation_period + time_to_test_delay)} day')


	#select second wave of infections
	startdate = '2020-07-01'
	df_daily_covid = df_daily_covid.loc[df_daily_covid.index > startdate]
	df_response = df_response.loc[df_response.index > startdate]

	print(df_daily_covid['Positivity_ratio'].tail())


	### make the plot
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	#third y axis for government response
	ax3 = ax1.twinx()
	ax3.spines['right'].set_position(('axes', 1.12))

	#plot number of positive tests
	lns1 = ax1.plot(df_daily_covid.index, df_daily_covid['Total_reported'],	label = 'Number of positive tests')
	#and the number of tests
	lns2 = ax1.plot(df_daily_covid.index, df_daily_covid['Number_of_tests'], label = 'Number of tests')
	#plot test positivity rate
	lns3 = ax2.plot(df_daily_covid[~df_daily_covid['Extrapolated']].index, df_daily_covid[~df_daily_covid['Extrapolated']]['Positivity_ratio']*100, label = 'Positivity rate', color = '#D10000')
	lns4 = ax2.plot(df_daily_covid[df_daily_covid['Extrapolated']].index, df_daily_covid[df_daily_covid['Extrapolated']]['Positivity_ratio']*100, label = 'Positivity rate (number of\ntests extrapolated)', color = '#D10000', linestyle = '--')

	#also plot government response
	lns5 = ax3.plot(df_response.index, df_response['StringencyIndex'], label = 'Stringency index', color = 'black')

	ax1.grid(linestyle = ':')

	lns = lns1 + lns2 + lns3 + lns4 + lns5
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc='best')

	ax1.set_xlabel(f'Estimated infection date\n(reporting date - incubation period (~6 days) - test delays (~{time_to_test_delay + result_delay} days))')
	ax1.set_ylabel('Number of tests per day')
	ax2.set_ylabel('Positivity rate [%]')
	ax3.set_ylabel('Oxford Stringency Index')

	ax1.set_title('SARS-CoV-2 tests in the Netherlands')

	# ax.xaxis.set_tick_params(rotation = 45)
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax1.xaxis.set_major_formatter(myfmt)

	ax1.set_ylim(0)
	ax2.set_ylim(0)

	plt.savefig(f'{plotloc}Tests_second_wave.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_hospitalization():
	"""

	"""
	df_IC = load_IC_data()

	print(df_IC)

	fig, ax = plt.subplots()

	ax.plot(df_IC.index, df_IC.New, label = 'New')
	ax.plot(df_IC.index, df_IC.Removed, label = 'Removed')

	ax.set_ylabel('Change')
	ax.set_title('Change in IC occupation')

	ax.legend(loc = 'best')
	ax.grid(linestyle = ':')

	fig.autofmt_xdate()

	plt.savefig(f'{plotloc}Hospital_IC_change.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_superspreader_events():
	df_SSE = load_superspreader_events_data()

	#plot distribution of the number of cases
	if False:
		fig, ax = plt.subplots()

		bins = np.linspace(np.nanmin(df_SSE['Total Cases']), np.percentile(df_SSE['Total Cases'].astype(float), 99), 100)

		ax.hist(df_SSE['Total Cases'], bins = bins)

		ax.grid(linestyle = ':')

		ax.set_xlabel('Number of infections $Z$')
		ax.set_ylabel('Number of SSE events')

		ax.set_title('Distribution of number of infections $Z$ at\nsuperspreader events (SSE)')

		plt.savefig(f'{plotloc}SSE/Distribution_number_of_cases_at_SSE.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

	#plot the occurences of SSEs at different settings
	if False:
		unique_settings, settings_count = np.unique(df_SSE['Setting1'], return_counts = True)
		sortloc = np.argsort(settings_count)[::-1]
		unique_settings = unique_settings[sortloc]
		settings_count = settings_count[sortloc]

		unique_settings[np.where(unique_settings == 'nan')[0][0]] = 'Unknown'

		#select those above 2 cases
		selection = settings_count > 10
		unique_settings_sel = unique_settings[selection]
		settings_count_sel = settings_count[selection]

		fig, ax = plt.subplots()

		bar_x = np.arange(len(settings_count_sel))

		ax.bar(bar_x, settings_count_sel)

		ax.set_xticks(bar_x)
		ax.set_xticklabels(unique_settings_sel, ha = 'right')
		ax.xaxis.set_tick_params(rotation = 75, labelsize = 9)

		ax.grid(linestyle = ':', axis = 'y')

		ax.set_ylabel('Number of SSE events')

		ax.set_title('Superspreader events settings (> 10 occurences)')

		plt.savefig(f'{plotloc}SSE/SSE_settings.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

	#get statistics on indoor/outdoor
	if True:
		unique_inout, inout_count = np.unique(df_SSE['Indoor / Outdoor'], return_counts = True)

		total_SSE = len(df_SSE)

		#replace unknown values
		nanloc = np.where(unique_inout == 'nan')[0]
		unknownloc = np.where(unique_inout == 'unknown')[0]
		Unknownloc = np.where(unique_inout == 'Unknown')[0][0]


		dellocs = list(nanloc) + list(unknownloc)
		if len(dellocs) > 0:
			if len(nanloc) > 0:
				inout_count[Unknownloc] += inout_count[nanloc[0]]
			if len(unknownloc) > 0:
				inout_count[Unknownloc] += inout_count[unknownloc[0]]

			unique_inout = np.delete(unique_inout, dellocs)
			inout_count = np.delete(inout_count, dellocs)

		# unique_settings[] = 'Unknown'

		for i in range(len(unique_inout)):
			print(f'{unique_inout[i]}: {inout_count[i]*100/total_SSE:0.01f}% ({inout_count[i]})')


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

def estimate_recent_R(enddate_train = '2020-10-25'):
	"""
	Estimate the reproductive number R with mobility data
	"""
	print(f'WARNING: end date for R training set is {enddate_train}')

	startdate_pred = '2020-07-01'

	df_prevalence, df_R = load_prevalence_R0_data()
	df_google_mob, df_apple_mob = load_mobility_data(smooth = True)
	df_sewage = load_sewage_data(smooth = True, shiftdates = True)

	#for the plot as a reference
	df_response = load_government_response_data()

	df_response = df_response.loc[df_response.index > startdate_pred]

	#determine error of R
	df_R['Rt_abs_error'] = ((df_R['Rt_low'] - df_R['Rt_avg']).abs() + (df_R['Rt_up'] - df_R['Rt_avg']).abs())/2

	#merge datasets
	# df_mob_R = df_google_mob.merge(df_R, right_index = True, left_index = True)
	df_mob_R = df_google_mob.join(df_R, how = 'outer')
	df_mob_R = df_mob_R.join(df_apple_mob, how = 'inner')
	df_mob_R = df_mob_R.join(df_sewage[['RNA_flow_smooth']], how = 'outer')

	#select date range
	mask = (df_mob_R.index > '2020-04-01') & (df_mob_R.index <= enddate_train)
	df_train = df_mob_R.loc[mask]
	df_pred = df_mob_R.loc[df_mob_R.index > startdate_pred]



	key_names = {
	'retail_recreation_smooth': 'Retail & recreation',
	'parks_smooth': 'Parks',
	'transit_stations_smooth': 'Transit stations',
	'workplaces_smooth': 'Workplaces',
	'residential_smooth': 'Residential',
	'RNA_flow_smooth': 'RNA flow smooth',
	'driving_smooth': 'Driving',
	'walking_smooth': 'Walking',
	'transit_smooth': 'Transit'
	}

	### plot correlation between the different mobility metrics and R
	if False:
		### plot results
		fig, axs = plt.subplots(ncols = 3, nrows = 3, sharex = False, sharey = 'row', figsize = (12, 12), gridspec_kw = {'hspace': 0.18, 'wspace': 0})
		axs = axs.flatten()

		for i, key in enumerate(key_names.keys()):
			#plot data
			axs[i].scatter(df_train[key], df_train['Rt_avg'], color = 'maroon', alpha = 0.6, s = 8, label = key_names[key])

			### fit a linear model
			popt, perr, r_squared = fit_model(linear_model,
								df_train[key],
								df_train['Rt_avg'],
								sigma = np.array(df_train['Rt_abs_error']))

			#plot the model
			xpoints = np.linspace(np.min(df_train[key]), np.max(df_train[key]), num = 500)
			axs[i].plot(xpoints, linear_model(xpoints, *popt), label = r'Fit ($R^2 = $' + f'{r_squared:0.03f})', color = 'black')

			axs[i].set_title(f'{key_names[key]}')

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

		plt.savefig(f'{mobplotloc}Mobility_R_correlation.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

	### combine the best correlating mobility metrics to predict R
	if True:
		best_correlating_metrics = [
			'retail_recreation_smooth',
			'transit_stations_smooth',
			'residential_smooth',
			'workplaces_smooth',
			'driving_smooth',
			'walking_smooth',
			'transit_smooth'
		]
		#get the multiple parameters into a single array
		# X_train = np.zeros((len(df_train), len(best_correlating_metrics)))
		# for i in range(len(best_correlating_metrics)):
		# 	X_train[:,i] = np.array(df_train[best_correlating_metrics[i]])
		X_train = dataframes_to_NDarray(df_train, best_correlating_metrics)
		Y_train = np.array(df_train['Rt_avg'])

		weight_train = 1/np.array(df_train['Rt_abs_error'])

		#apply ridge regression, see here for more info:
		#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
		clf = Ridge(alpha = 1.)
		# clf = Lasso()
		#apply adaboost regression, see here for more info:
		#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
		# clf = AdaBoostRegressor()
		# clf = RandomForestRegressor()
		clf.fit(X_train, Y_train, sample_weight = weight_train)
		# clf.fit(X_train, Y_train)

		r_squared = clf.score(X_train, Y_train, sample_weight = weight_train)
		# r_squared = clf.score(X_train, Y_train)

		print(f'R^2: {r_squared:0.03f}')

		### plot the predictions versus ground truth
		fig, ax = plt.subplots()

		ax.scatter(Y_train, clf.predict(X_train), alpha = 0.4, color = 'navy', s = 8, label = f'Predictions ($R^2$ = {r_squared:0.03f})')
		#indicate one to one relationship
		xlims = ax.get_xlim()
		ylims = ax.get_ylim()

		startpoint = min((min(xlims), min(ylims)))
		endpoint = max((max(xlims), max(ylims)))

		ax.plot([startpoint, endpoint], [startpoint, endpoint], color = 'black', label = 'Ideal predictions')
		ax.set_xlim(xlims)
		ax.set_ylim(ylims)

		ax.grid(linestyle = ':')
		ax.legend(loc = 'best')

		ax.set_xlabel('Measured R')
		ax.set_ylabel('Predicted R')

		ax.set_title('R prediction accuracy using mobility data')

		plt.savefig(f'{mobplotloc}Mobility_R_prediction_accuracy.png', dpi = 200, bbox_inches = 'tight')
		plt.close()


		### now make and plot predictions
		X_pred = dataframes_to_NDarray(df_pred, best_correlating_metrics)

		Y_pred = clf.predict(X_pred)


		fig, ax1 = plt.subplots()

		ax2 = ax1.twinx()

		ln1 = ax1.plot(df_pred.index, df_pred.Rt_avg, label = 'Ground truth', color = betterblue)
		#indicate error margins on ground truth
		ax1.fill_between(df_pred.index, df_pred['Rt_low'], df_pred['Rt_up'], alpha = 0.4, color = betterblue)

		ln2 = ax1.plot(df_pred.index, Y_pred, label = f'Prediction ($R^2$: {r_squared:0.03f})', color = betterorange)

		#also plot government response
		ln3 = ax2.plot(df_response.index, df_response['StringencyIndex'], label = 'Stringency index', color = betterblack)

		ax1.grid(linestyle = ':')
		# ax.legend(loc = 'best')

		lns = ln1 + ln2 + ln3
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc = 'lower left')

		ax1.set_ylabel('$R$')
		ax2.set_ylabel('Oxford Stringency Index')
		ax1.set_title('$R$ prediction')

		ax2.set_ylim(0)

		#fix date ticks
		ax1.set_xticks(pd.date_range(np.min(df_pred.index), np.max(df_pred.index + pd.Timedelta(f'7 day')), freq = '2W'))
		fig.autofmt_xdate()
		myfmt = mdates.DateFormatter('%d-%m-%Y')
		ax1.xaxis.set_major_formatter(myfmt)

		plt.savefig(f'{mobplotloc}Mobility_R_prediction.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

def estimate_recent_prevalence(enddate_train = '2020-11-01', smoothsize = 5):
	"""
	Estimate the recent prevalence based on the test positivity ratio
	"""
	print(f'WARNING: end date for prevalence training set is {enddate_train}')

	startdate_train = '2020-09-08'

	df_daily_covid = load_daily_covid(correct_for_delay = False)
	df_n_tests = load_number_of_tests()
	df_prevalence, df_R0 = load_prevalence_R0_data()
	df_sewage = load_sewage_data(smooth = True, shiftdates = False)
	#for the plot as a reference
	df_response = load_government_response_data()

	df_response = df_response.loc[df_response.index > startdate_train]

	### correct for the delay between the onset of symptoms and the result of the test
	#for source of incubation period, see the readme
	incubation_period = 6 #days, left of average of 8.3 due to extremely skewed distribution
	#test delay determined from anecdotal evidence
	time_to_test_delay = 2 #days
	result_delay = 1

	#correct the results to the day of the test
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{result_delay} day')

	#merge datasets
	df_daily_covid = df_daily_covid.merge(df_n_tests[['Number_of_tests']], right_index = True, left_index = True)

	#determine test positivity rate
	df_daily_covid['Positivity_ratio'] = df_daily_covid['Total_reported']/df_daily_covid['Number_of_tests']

	#smooth the testing data, as there are some days with IT problems and underreported
	#tests, while having overreports the next few days.
	if smoothsize != None:
		df_daily_covid['Positivity_ratio_smooth'] = df_daily_covid['Positivity_ratio'].rolling(smoothsize).mean()

		print('Using smoothed positivity data')

	#correct the results to the day of infection
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{int(incubation_period + time_to_test_delay)} day')

	#determine error on prevalence
	df_prevalence['prev_abs_error'] = ((df_prevalence['prev_low'] - df_prevalence['prev_avg']).abs() + (df_prevalence['prev_up'] - df_prevalence['prev_avg']).abs())/2

	df_daily_covid['Total_per_million'] = df_daily_covid['Total_reported'] * per_million_factor

	#merge testing and sewage data
	df_predictors = df_daily_covid.merge(df_sewage[['RNA_flow_smooth']], right_index = True, left_index = True)

	#select second wave of infections with the high test rate, but stop at the
	#section where the prevalence flattens (seems unrealistic)
	#need to select after 2020-09-06 because that's when the sewage measurements change to per 100.000
	df_predictors_sel = df_predictors.loc[(df_predictors.index > startdate_train) & (df_predictors.index < enddate_train)]
	df_prevalence_sel = df_prevalence.loc[(df_prevalence.index > startdate_train) & (df_prevalence.index < enddate_train)]


	#discard days with too low number of positive tests per million for determining the correlation
	# test_pos_threshold = 40

	# startdate_for_cor = np.argmax(df_predictors_sel['Total_per_million'] > test_pos_threshold)
	startdate_for_cor = startdate_train
	#datasets for determining correlation
	df_predictors_cor = df_predictors_sel.loc[df_predictors_sel.index > startdate_for_cor]
	df_prevalence_cor = df_prevalence_sel.loc[df_prevalence_sel.index > startdate_for_cor]

	parameters_used = [
	'RNA_flow_smooth'
	]

	if smoothsize != None:
		parameters_used.append('Positivity_ratio_smooth')
	else:
		parameters_used.append('Positivity_ratio')

	'''
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()

	ax1.plot(df_predictors_cor.index, df_predictors_cor['Test_pos_ratio'], color = 'maroon')
	ax2.plot(df_prevalence_cor.index, df_prevalence_cor['prev_avg'])

	plt.savefig('test.png')
	plt.close()
	'''

	### get data into the shape required for sklearn functions
	X_train = dataframes_to_NDarray(df_predictors_cor, parameters_used)

	Y_train = np.array(df_prevalence_cor['prev_avg'])
	weight_train = 1/np.array(df_prevalence_cor['prev_abs_error'])

	### apply regression
	# clf = Ridge(alpha = 1.)
	clf = LinearRegression()
	# clf = AdaBoostRegressor()
	clf.fit(X_train, Y_train, sample_weight = weight_train)

	r_squared = clf.score(X_train, Y_train, sample_weight = weight_train)

	print(f'R^2: {r_squared:0.03f}')


	### plot accuracy of predictions using the predictions versus ground truth
	fig, ax = plt.subplots()

	train_pred = clf.predict(X_train)
	ax.scatter(Y_train, train_pred, alpha = 0.4, color = 'navy', s = 8, label = f'Predictions ($R^2$ = {r_squared:0.03f})')

	#plot error bars
	errorbar_data = np.stack((df_prevalence_cor['prev_low'].values, df_prevalence_cor['prev_up'].values))
	ax.errorbar(Y_train, train_pred, xerr = errorbar_data, ecolor = 'navy', elinewidth = 0.5, capthick = 0.5, capsize = 2, errorevery = 2, ls = 'none')


	#indicate one to one relationship
	xlims = ax.get_xlim()
	ylims = ax.get_ylim()

	startpoint = min((min(xlims), min(ylims)))
	endpoint = max((max(xlims), max(ylims)))

	ax.plot([startpoint, endpoint], [startpoint, endpoint], color = 'black', label = 'Ideal predictions (linear model)')
	ax.set_xlim(xlims)
	ax.set_ylim(ylims)

	ax.grid(linestyle = ':')
	ax.legend(loc = 'best')

	ax.set_xlabel('Measured prevalence [per million]')
	ax.set_ylabel('Predicted prevalence [per million]')

	ax.set_title('Prevalence prediction accuracy using mobility data')

	plt.savefig(f'{plotloc}Prevalence_prediction_accuracy.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


	### Now make predictions for the most recent data
	#select the positive test data
	df_predictors_pred = df_predictors.loc[df_predictors.index > startdate_for_cor]
	#get data in sklearn shape
	Xpred = dataframes_to_NDarray(df_predictors_pred, parameters_used)
	#make the predictions
	# df_predictors_pred['Prev_pred'] = linear_model(df_daily_covid_pred['Total_per_million'], *popt)
	df_predictors_pred['Prev_pred'] = clf.predict(Xpred)

	df_prevalence_pred = df_predictors_pred[['Prev_pred']]


	###now plot together with the old data
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()

	#old data
	ln1 = ax1.plot(df_prevalence_sel.index, df_prevalence_sel['prev_avg'], label = 'Measurements', color = betterblue)
	#show error bars
	ax1.fill_between(df_prevalence_sel.index, df_prevalence_sel['prev_low'], df_prevalence_sel['prev_up'], alpha = 0.4, color = betterblue)

	#predictions
	ln2 = ax1.plot(df_prevalence_pred.index, df_prevalence_pred['Prev_pred'], label = f'Linear prediction ($R^2 = {r_squared:0.03f}$)', color = betterorange)

	#also plot government response
	ln3 = ax2.plot(df_response.index, df_response['StringencyIndex'], label = 'Stringency index', color = betterblack)

	ax1.grid(linestyle = ':')

	ax1.set_ylabel('Prevalence per million')
	ax2.set_ylabel('Oxford Stringency Index')

	ax1.set_title('COVID-19 estimated active cases in the Netherlands\nwith prediction of recent days')

	lns = ln1 + ln2 + ln3
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc = 'lower right')

	# ax.xaxis.set_tick_params(rotation = 45)
	# ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	#set location of date ticks, as the automatic technique does not work
	ax1.set_xticks(pd.date_range(np.min(df_prevalence_sel.index), np.max(df_prevalence_pred.index), freq = 'W'))

	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax1.xaxis.set_major_formatter(myfmt)

	ax1.set_ylim(0)
	ax2.set_ylim(0)

	plt.savefig(f'{plotloc}Prevalence_second_wave_with_predictions.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def main():
	# government_response_results_simple()

	# plot_superspreader_events()

	estimate_recent_R(enddate_train = '2020-10-28')

	# estimate_recent_prevalence(enddate_train = '2020-11-01')

	# plot_prevalence_R()

	# plot_mobility()

	# plot_daily_results()

	# plot_sewage()

	# plot_hospitalization()

	# stringency_R_correlation()


if __name__ == '__main__':
	main()
#
