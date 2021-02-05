import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.patches as mpatches
from turbo_colormap import turbo_colormap_data
turbo_cmap = mcolors.ListedColormap(turbo_colormap_data)

import urllib.request
from urllib.error import HTTPError
import shutil
import os, sys, glob
from zipfile import ZipFile
from tqdm import tqdm

import datetime as dt

from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

from astropy import units as u

dataloc = './Data/'
static_data_loc = f'{dataloc}Edit_only/'

plotloc = './Plots/'
plotloc_government_response = './Gov_response_plots/'
R_plotloc = f'{plotloc}R_prediction/'
Prev_plotloc = f'{plotloc}Prevalence_prediction/'
epidem_modelling_plotloc = f'{plotloc}Epidemiological_modelling/'

#source: https://www.cbs.nl/nl-nl/visualisaties/bevolkingsteller
n_inhabitants_NL = 17455552
per_million_factor = 1e6/n_inhabitants_NL

#number of days after which an infected person becomes infectuous themselves
serial_interval = 5.

#number of days between infection and recovery
time_till_recovery = 14

betterblue = '#016FB9'
betterorange = '#F17105'
betterblack = '#141414'
mplblue = 'steelblue'

### dateshift values needed to correct for the delay between the onset of
### symptoms and the result of the test
#for source of incubation period, see the readme
incubation_period = 6 #days, left of average of 8.3 due to extremely skewed distribution
#delay between first symptoms (date of desease onset) and test performed, determined
#by willingness to be tested and available testing capacity. Set by assumption
time_to_test_delay = 2 #days
result_delay = 1

def downloadSave(url, file_name, check_file_exists = False):
	"""
	Return True when a new file is downloaded
	"""
	if not os.path.exists(file_name):
		print('Downloading file...')
		with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
			shutil.copyfileobj(response, out_file)

		return True

def average_kernel(size = 7):
	return np.ones(size)/size

def exponential_model(nE_0, Rt, t, tau):
	return nE_0 * Rt**(t/tau)

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

def get_period_dates(periodtype):
	"""
	Get date ranges of certain important periods, like school closures, vacations etc
	"""

	if periodtype.lower() == 'autumn break':
		return np.array(['2020-10-10', '2020-10-24'], dtype = np.datetime64)
	elif periodtype.lower() == 'school closure':
		return np.array(['2020-03-20', '2020-08-24'], dtype = np.datetime64)
	else:
		raise ValueError('Incorrect period type given')

def shade_region(ax, dates, colour = 'black', alpha = 0.2, label = None):
	"""
	Shade a region in a matplotlib graph
	"""
	xlims = ax.get_xlim()
	ylims = ax.get_ylim()

	ln = ax.fill([dates[0], dates[1], dates[1], dates[0]],
			[ylims[0], ylims[0], ylims[1], ylims[1]],
			alpha = alpha, edgecolor = 'none', facecolor = colour, label = label)

	ax.set_xlim(xlims)
	ax.set_ylim(ylims)

	return ln

def indicate_school_closed(ax):
	autumn_break_dates = get_period_dates('autumn break')
	shade_region(ax, autumn_break_dates, label = 'School closure/vacations')
	school_closure_dates = get_period_dates('school closure')
	shade_region(ax, school_closure_dates)

def indicate_incomplete_test_data(ax, df):
	"""
	Indicate the latest few days where test data for the individual dataset
	is as of yet incomplete
	"""
	return shade_region(ax, np.array([df.index[-1] - pd.Timedelta(f'{5} day'), df.index[-1]], dtype = np.datetime64), colour = 'red', alpha = 0.2, label = 'Data incomplete')

def extrapolate_dataframe(df, colname, date_to_extrap, base_period = 7):
	"""
	Extrapolate data in a dataframe with a date index

	date_to_extrap (np.datetime64): date up to which you want to extrapolate to
	"""
	#indicate which part was extrapolated
	df['Extrapolated'] = 0

	#do not extrapolate if the date to extrapolate to is smaller than the final date
	if not (date_to_extrap - np.datetime64(df.index[-1]) > np.timedelta64(23, 'h')) and not (np.datetime64(df.index[0]) - date_to_extrap > np.timedelta64(23, 'h')):
		df = df.astype({'Extrapolated': 'bool'})

		return df

	#check if we extrapolate on the front or back of the data sequence
	extrap_front = False
	if date_to_extrap < np.datetime64(df.index[0]):
		extrap_front = True

	#select on which section we want to extrapolate (up to 3 weeks back)
	if extrap_front:
		start_extrapolation_pred = df.index[0]
		start_extrapolation = start_extrapolation_pred + pd.Timedelta(f'{base_period} day')
		df_extrap_train = df.loc[df.index < start_extrapolation]
	else:
		start_extrapolation_pred = df.index[-1]
		start_extrapolation = start_extrapolation_pred - pd.Timedelta(f'{base_period} day')
		df_extrap_train = df.loc[df.index > start_extrapolation]

	#get a numerical index in epoch time
	X_train = df_extrap_train.index.astype('int64')
	Y_train = df_extrap_train[colname].values

	## train model
	clf = LinearRegression()
	clf.fit(X_train[:,None], Y_train)

	## now make the proper array
	#init the enddate
	df.loc[date_to_extrap] = [np.nan]*(len(df.columns) - 1) + [1]
	#add dates in between
	df = df.resample('1d').mean()

	#extract the rows for extrapolation
	if extrap_front:
		df_extrap_pred = df.loc[df.index < start_extrapolation]
		selectindex = (df_extrap_pred.index < start_extrapolation_pred)
	else:
		df_extrap_pred = df.loc[df.index > start_extrapolation]
		selectindex = (df_extrap_pred.index > start_extrapolation_pred)

	#get numerical index
	X_pred = df_extrap_pred.index.astype('int64')[selectindex]

	#re-select index
	if extrap_front:
		selectindex = (df_extrap_pred.index < start_extrapolation_pred)
	else:
		selectindex = (df_extrap_pred.index > start_extrapolation_pred)

	#get predictions
	df_temp = pd.DataFrame(data = clf.predict(X_pred[:,None]), index = df_extrap_pred.loc[selectindex].index, columns = [colname])
	#indicate that these values are extrapolated
	df_temp['Extrapolated'] = 1

	#now update the interpolated array
	df.update(df_temp)

	df = df.astype({'Extrapolated': 'bool'})

	return df

def rel_humidity_conversion(RH, T, Ptot = None, humidity_type = 'absolute'):
	"""
	Based on the humidity conversion formulas by Vaisala

	Input:
	relative humidity (RH, in percent)
	temperature (T, in degrees celcius)
	Ptot: ambient pressure
	"""
	T = T.to(u.K, equivalencies = u.temperature())

	### constants
	#parameters for critical point of water, see https://en.wikipedia.org/wiki/Critical_point_(thermodynamics)
	Tc = 647.096 * u.K
	Pc = 22.064 * u.MPa

	#coefficients from Wagner and Pruss, 1995
	C1 = -7.85951783
	C2 = 1.84408259
	C3 = -11.7866497
	C4 = 22.6807411
	C5 = -15.9618719
	C6 = 1.80122502

	C = 2.16679 * u.g * u.K / u.J

	#mixing ratio constant for air
	B = 621.9907 *u.g/u.kg


	### equations
	#eq 2
	v = 1 - T/Tc
	#eq 3
	Pws = Pc * np.exp((Tc/T) * (C1*v + C2*v**1.5 + C3*v**3 + C4*v**3.5 + C5*v**4 + C6*v**7.5))
	#eq 1
	Pw = (RH/100) * Pws

	if humidity_type == 'absolute':
		#eq 17
		AH = C * Pw/T

		return AH.decompose()
	elif humidity_type == 'mixing ratio':
		if Ptot is None:
			raise ValueError('No ambient pressure given')
		#eq 14
		X = B*Pw/(Ptot - Pw)

		return X/(u.g/u.kg)
	else:
		raise ValueError('No proper desired humidity type given')

def whitening_transform(X, return_W_matrix = False):
	"""
	Decorrelate input data parameters by performing a whitening transformation
	https://en.wikipedia.org/wiki/Whitening_transformation

	Based on:
	https://learndataa.com/2020/09/15/data-preprocessing-whitening-or-sphering-in-python/
	"""

	# Center data
	# By subtracting mean for each feature
	Xc = X - np.mean(X, axis = 0)
	Xc = Xc.T

	# Calculate covariance matrix
	Xcov = np.cov(Xc, bias = True)

	# Calculate Eigenvalues and Eigenvectors
	w, v = np.linalg.eig(Xcov)
	# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits

	# Calculate inverse square root of Eigenvalues
	# Optional: Add '.1e5' to avoid division errors if needed
	# Create a diagonal matrix
	diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
	diagw = diagw.real.round(4) #convert to real and round off

	# Calculate Rotation (optional)
	# Note: To see how data can be rotated
	# Xrot = np.dot(v, Xc)

	# Whitening transform using PCA (Principal Component Analysis)
	# Wpca = np.dot(np.dot(diagw, v.T), Xc).T
	Wpca = diagw @ v.T
	Xwhitened = (Wpca @ Xc).T

	# Whitening transform using ZCA (Zero Component Analysis)
	# wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)

	if return_W_matrix:
		return Xwhitened, Wpca
	else:
		return Xwhitened

def load_prevalence_Rt_data():
	"""
	Load data and return as estimated number of infectious people per million
	"""
	url_prevalence = 'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json'
	url_Rt = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

	fname_prevalence = f'{dataloc}prevalence.json'
	fname_Rt = f'{dataloc}Rt.json'

	downloadSave(url_prevalence, fname_prevalence, check_file_exists = True)
	downloadSave(url_Rt, fname_Rt, check_file_exists = True)

	df_prevalence = pd.read_json(fname_prevalence)[['Date', 'prev_avg', 'prev_low',
	'prev_up']]
	df_Rt = pd.read_json(fname_Rt)[['Date', 'Rt_avg', 'Rt_low', 'Rt_up']]

	df_prevalence.set_index('Date', inplace = True)
	df_Rt.set_index('Date', inplace = True)

	#change prevalence to per million
	df_prevalence['prev_avg'] *= per_million_factor
	df_prevalence['prev_low'] *= per_million_factor
	df_prevalence['prev_up'] *= per_million_factor

	return df_prevalence, df_Rt

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

def load_government_response_data(country = 'NLD'):
	"""
	More info on the various flags:
	https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md
	"""
	url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
	raw_fname = f'{dataloc}OxCGRT_latest.csv'

	#filename of the csv data containing only the desired dutch data
	nl_fname = f'{dataloc}OxCGRT_{country}.csv'

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
		df_response = df_response.loc[df_response['CountryCode'] == country]

		#get the individual index column names
		desired_indices = np.array([f'C{x}' for x in range(1, 9)] + ['E1', 'E2'])
		all_columns = np.array(list(df_response.columns))
		desired_indices_names = {all_columns[np.flatnonzero(np.core.defchararray.find(all_columns, idx) != -1)[0]]: idx for idx in desired_indices}

		#select desired columns
		df_response = df_response[['Date', 'StringencyIndex'] + list(desired_indices_names.keys())]

		#rename columns to shorter names
		df_response = df_response.rename(columns = desired_indices_names)

		#read dates
		df_response['Date'] = pd.to_datetime(df_response['Date'], format = '%Y%m%d')
		df_response.set_index('Date', inplace = True)

		###extrapolate the last few values
		#determine nan locs
		nanloc = df_response.loc[df_response['StringencyIndex'].isnull()].index
		#and last date of not nan
		last_date_with_data = df_response.loc[~df_response['StringencyIndex'].isnull()].index[-1]

		#now fill the rows with no data with the latest data
		for l in nanloc:
			df_response.loc[l] = df_response.loc[last_date_with_data]

		# save this edited file to a csv
		df_response.to_csv(nl_fname)

	return df_response

def load_mobility_data(smooth = False, smoothsize = 7, apple_mobility_url_base = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2025HotfixDev21/v3/en-us/applemobilitytrends-'):
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
		for minday in range(14):
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
	fname_daily_covid = f'{dataloc}covid19_postests_hosp_death.csv'

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

def load_number_of_tests(enddate = None, ignore_last_datapoint = False):
	"""
	Data source:
	https://www.rivm.nl/documenten/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland

	Compiled into a .csv by hand
	"""

	df_n_tests = pd.read_csv(f'{dataloc}Edit_only/tests_per_week.csv', usecols = ['Year', 'Week_number', 'Number_of_tests'])

	df_n_tests = df_n_tests.astype({'Week_number': str})

	#get date from week number, assume the central day of the week (thursday)
	df_n_tests['Date'] = pd.to_datetime(df_n_tests['Week_number'].astype(str) + '-4-' + df_n_tests['Year'].astype(str), format = '%W-%w-%Y')

	df_n_tests.set_index('Date', inplace = True)

	del df_n_tests['Week_number']

	### interpolate to each day
	#first convert units to per day
	df_n_tests['Number_of_tests'] /= 7

	#and ignore last known date (due to incomplete data) if given a final date
	if ignore_last_datapoint and enddate != None:
		df_n_tests = df_n_tests[:-1]

	#resample to once a day
	df_n_tests = df_n_tests.resample('1d').mean()

	#copy the column for inspection of the interpolation results
	df_n_tests['Number_of_tests_raw'] = df_n_tests['Number_of_tests']

	#then interpolate/extrapolate
	df_n_tests['Number_of_tests'] = df_n_tests['Number_of_tests'].interpolate(method = 'linear')

	#now if the enddate is given, we also want to extrapolate
	if enddate != None:
		df_n_tests = extrapolate_dataframe(df_n_tests, 'Number_of_tests', enddate, base_period = 7)

	return df_n_tests

def load_sewage_data(smooth = False, windowsize = 3, shiftdates = False):
	sewage_url = 'https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv'
	sewage_fname = f'{dataloc}COVID-19_rioolwaterdata.csv'

	downloadSave(sewage_url, sewage_fname, check_file_exists = True)

	df_sewage = pd.read_csv(sewage_fname, usecols = ['Date_measurement', 'Security_region_name', 'Percentage_in_security_region', 'RNA_flow_per_100000', 'Representative_measurement'])

	#shift RNA units to "* 100 billion"
	df_sewage['RNA_flow_per_100000'] /= 100e9

	#only take "representative measurements" which span 24 hours instead of a single moment
	#remove nans
	df_sewage = df_sewage.loc[df_sewage['Representative_measurement'].notna()]
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

def load_IC_data(get_change = False):
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

	if get_change:
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
	else:
		return df_IC_count

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

def load_pop_pyramid():
	"""
	Data source:
	https://opendata.cbs.nl/statline/?dl=308BE#/CBS/nl/dataset/7461bev/table
	"""
	fname_pop_pyramid = f'{static_data_loc}Bevolking__leeftijd__burgerlijke_staat_1-1-2020.csv'
	df_pop_pyramid = pd.read_csv(fname_pop_pyramid, usecols = ['Geslacht', 'Leeftijd', 'Bevolking (aantal)', 'Burgerlijke staat'], sep = ';')

	#only use columns of totals, ignore sex
	df_pop_pyramid = df_pop_pyramid.loc[df_pop_pyramid['Geslacht'] == 'Totaal mannen en vrouwen']
	#only select the totals of the maritial status
	df_pop_pyramid = df_pop_pyramid.loc[df_pop_pyramid['Burgerlijke staat'] == 'Totaal burgerlijke staat']

	del df_pop_pyramid['Geslacht'], df_pop_pyramid['Burgerlijke staat']

	df_pop_pyramid = df_pop_pyramid.rename(columns = {'Leeftijd': 'Agegroup', 'Bevolking (aantal)': 'N_people'})

	df_pop_pyramid.set_index('Agegroup', inplace = True)

	#ignore total
	df_pop_pyramid = df_pop_pyramid.loc[df_pop_pyramid.index != 'Totaal leeftijd']

	df_pop_pyramid_translated = pd.DataFrame(columns = ['Agegroup', 'N_people'])
	#go from 5 year to 10 year groups and translate to '0-9', with a max of '90+'
	for agesteps in np.arange(10, 101, 10):
		if agesteps <= 90:
			total_10_year_agegroup = int(df_pop_pyramid.loc[f'{agesteps-10} tot {agesteps-5} jaar'] + df_pop_pyramid.loc[f'{agesteps-5} tot {agesteps} jaar'])

			df_pop_pyramid_translated = df_pop_pyramid_translated.append({'Agegroup': f'{agesteps-10}-{agesteps-1}', 'N_people': total_10_year_agegroup}, ignore_index = True)
		else:
			total_10_year_agegroup = int(df_pop_pyramid.loc[f'{agesteps-10} tot {agesteps-5} jaar'] + df_pop_pyramid.loc[f'{agesteps-5} tot {agesteps} jaar'] + df_pop_pyramid.loc[f'100 jaar of ouder'])

			df_pop_pyramid_translated = df_pop_pyramid_translated.append({'Agegroup': '90+', 'N_people': total_10_year_agegroup}, ignore_index = True)

	df_pop_pyramid_translated.set_index('Agegroup', inplace = True)

	return df_pop_pyramid_translated

def load_individual_positive_test_data(load_agegroups = False, load_datatype = 'pos_test', correct_for_pop_pyramid = True):
	"""
	Load data on individual cases of positive tests of COVID-19, including data
	such as age, and date of disease onset/positive lab result/GGD notification,
	whichever came first.

	More info:
	https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/2c4357c8-76e4-4662-9574-1deb8a73f724

	Legend:
	DOO = Date of disease onset
	DPL = Date of first Positive Labresult
	DON = Date of Notification
	"""
	individual_data_url = 'https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv'
	individual_data_fname = f'{dataloc}individual_cases.csv'

	downloadSave(individual_data_url, individual_data_fname, check_file_exists = True)

	#load only a few columns
	df_individual = pd.read_csv(individual_data_fname, usecols = ['Date_statistics', 'Date_statistics_type', 'Agegroup', 'Hospital_admission', 'Deceased'], sep = ';')

	df_individual = df_individual.rename(columns = {'Date_statistics': 'Date'})
	df_individual['Date'] = pd.to_datetime(df_individual['Date'], format = '%Y-%m-%d')

	df_individual.set_index('Date', inplace = True)

	### We want all dates to be DOO ideally. To do this, we will convert DPL and
	### DON dates using estimates
	#first DON, biggest shift back
	df_individual.loc[df_individual['Date_statistics_type'] == 'DON'].index = df_individual.loc[df_individual['Date_statistics_type'] == 'DON'].index - pd.Timedelta(f'{time_to_test_delay + result_delay} day')
	#then DPL, smaller shift
	df_individual.loc[df_individual['Date_statistics_type'] == 'DPL'].index = df_individual.loc[df_individual['Date_statistics_type'] == 'DPL'].index - pd.Timedelta(f'{time_to_test_delay} day')

	#now we can get rid of the date statistic type column
	del df_individual['Date_statistics_type']

	if load_agegroups:
		#if we only want to look at deaths, remove all entries which do not conceirn
		#deceased people
		if load_datatype == 'deaths':
			df_individual = df_individual.loc[df_individual['Deceased'] == 'Yes']
		elif load_datatype == 'hosp':
			df_individual = df_individual.loc[df_individual['Hospital_admission'] == 'Yes']

		del df_individual['Deceased'], df_individual['Hospital_admission']

		#we also want an indication of the distribution of the different age groups,
		#so we first get them in different columns
		all_agegroups = np.sort(df_individual['Agegroup'].unique())
		df_individual_agegroups = pd.get_dummies(df_individual['Agegroup']).reindex(columns = all_agegroups, fill_value = 0)
		#then aggregate
		df_individual_agegroups = df_individual_agegroups.groupby(['Date']).agg(dict(zip(all_agegroups, ['sum']*len(all_agegroups))))

		#remove pesky columns like '<50' and 'unknown'
		if load_datatype != 'pos_test':
			removecols = ['Unknown']
		else:
			removecols = ['<50', 'Unknown']
		for removecol in removecols:
			if removecol in df_individual_agegroups.columns:
				del df_individual_agegroups[removecol]

		if correct_for_pop_pyramid:
			### correct for population pyramid
			#load data
			df_pop_pyramid = load_pop_pyramid()

			#run over the columns and convert to a fraction of the whole population in
			#that age group
			for col in df_individual_agegroups.columns:
				df_individual_agegroups[col] /= float(df_pop_pyramid.loc[col])

			#add dates with no cases
			df_individual_agegroups = df_individual_agegroups.resample('1d').asfreq(fill_value = 0)

		return df_individual_agegroups
	else:
		del df_individual['Deceased']
		#we need a new column for the number of cases
		df_individual.loc[:,'N_cases'] = 1

		#now aggregate over the dates
		df_individual_aggr = df_individual.groupby(['Date']).agg({'N_cases': 'sum'})

		#add dates with no cases
		df_individual_aggr = df_individual_aggr.resample('1d').asfreq(fill_value = 0)

		return df_individual_aggr

def load_cluster_data():
	"""
	Load data on COVID-19 clusters from
	https://coronalocator.knack.com/corona-locator#home/
	"""

	df_clusters = pd.read_csv(f'{static_data_loc}coronalocator_clusters.csv', usecols = ['Datum', 'Gemeente', 'Betreft', 'Aantal COVID-19'])

	df_clusters = df_clusters.rename(columns = {'Datum': 'Date', 'Betreft': 'Setting', 'Aantal COVID-19': 'N_infections'})

	df_clusters['Date'] = pd.to_datetime(df_clusters['Date'], format = '%d/%m/%Y ')

	### we will group by week for the data visualization
	#first shift dates by 7 days so it represents the start of the week
	df_clusters['Date'] -= pd.to_timedelta(7, unit = 'd')
	df_clusters.set_index('Date', inplace = True)

	#add column indicating the number of clusters
	# df_clusters.loc[:,'N_clusters'] = 1

	#perform the group by week
	# df_clusters = df_clusters.groupby(['Setting', pd.Grouper(key = 'Date', freq = 'W-MON')])['N_clusters'].sum().reset_index().sort_values('Date')

	#give each unique setting a column
	all_settings = np.sort(df_clusters['Setting'].unique())
	df_clusters = pd.get_dummies(df_clusters['Setting']).reindex(columns = all_settings, fill_value = 0)

	#then group by week
	df_clusters = df_clusters.groupby([pd.Grouper(freq = 'W-MON')])[all_settings].sum().reset_index().sort_index()

	df_clusters.set_index('Date', inplace = True)

	return df_clusters

def load_weather_data(smooth = False, abs_hum = False):
	"""
	Load weather data from the KNMI, location De Bilt. More info:
	https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
	"""

	url_weather = 'https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip'
	fname_weather_zip = f'{dataloc}daily_weather.zip'
	fname_weather = f'{dataloc}daily_weather.txt'

	downloaded = downloadSave(url_weather, fname_weather_zip, check_file_exists = True)

	if downloaded:
		#unzip
		unziploc = f'{dataloc}unzip/'
		with ZipFile(fname_weather_zip, 'r') as zfile:
			os.mkdir(unziploc)
			zfile.extractall(unziploc)

		#get the desired file
		zipfilename = glob.glob(unziploc + '*.txt')[0]
		# print(zipfilename)
		# #rename
		os.rename(zipfilename, fname_weather)

		#remove downloaded files
		shutil.rmtree(unziploc)
		#also remove zip file
		# os.remove(google_mobility_fname_zip)

	df_weather = pd.read_csv(fname_weather, skiprows = 47, usecols = ['YYYYMMDD', '   TG', '	Q', '   UG', '   PG'])

	df_weather = df_weather.rename(columns = {'YYYYMMDD': 'Date', '   TG': 'TAvg', '	Q': 'Rad', '   UG': 'HumAvg', '   PG': 'P'})

	df_weather['Date'] = pd.to_datetime(df_weather['Date'], format = '%Y%m%d')
	df_weather.set_index('Date', inplace = True)

	#select data from start of the pandemic
	df_weather = df_weather.loc[df_weather.index > '2020-02-15']

	#convert average temperature from 0.1 degrees C to 1 degree C
	df_weather['TAvg'] /= 10

	#calculate the absolute humidity (sometimes known as mixing ratio)
	df_weather['HumAbsAvg'] = rel_humidity_conversion(df_weather['HumAvg'].values, df_weather['TAvg'].values*u.deg_C, Ptot = df_weather['P'].values*10*u.Pa, humidity_type = 'mixing ratio').decompose()

	#smooth data if desired
	if smooth:
		windowsize = 3

		df_weather = df_weather.rolling(windowsize).mean()

	# fig, ax = plt.subplots()
	# ax.plot(df_weather.index, df_weather['Rad'])
	# plt.show()

	return df_weather

def load_behaviour_data(startdate, enddate):
	"""
	Load data on questionnaires about how much people follow the COVID-19 rules.
	https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/8a72d78a-fcf8-4882-b0ab-cd594961a267?tab=relations

	startdate, enddate as np.datetime64.
	"""
	behaviour_url = 'https://data.rivm.nl/covid-19/COVID-19_gedrag.csv'
	behaviour_fname = 'COVID-19_gedrag.csv'

	downloadSave(behaviour_url, behaviour_fname, check_file_exists = True)

	loadcols = [
	'Date_of_measurement',
	'Region_name',
	'Subgroup_category',
	'Indicator_category',
	'Indicator',
	'Figure_type',
	'Value'#,
	# 'Lower_limit',
	# 'Upper_limit'
	]

	df_behaviour = pd.read_csv(behaviour_fname, sep = ';', usecols = loadcols)

	df_behaviour = df_behaviour.rename(columns = {'Date_of_measurement': 'Date'})

	df_behaviour['Date'] = pd.to_datetime(df_behaviour['Date'], format = '%Y-%m-%d %H:%M:%S')
	df_behaviour.set_index('Date', inplace = True)

	### start filtering out undesired rows
	#select only country wide
	df_behaviour = df_behaviour.loc[df_behaviour.Region_name == 'Nederland']
	#select all subgroups
	df_behaviour = df_behaviour.loc[df_behaviour.Subgroup_category == 'Alle']
	#select on data on if people follow the rules
	df_behaviour = df_behaviour.loc[df_behaviour.Indicator_category == 'Naleving']
	#only want percentages
	df_behaviour = df_behaviour.loc[df_behaviour.Figure_type == 'Percentage']

	#remove the filter columns
	df_behaviour = df_behaviour.drop(['Figure_type', 'Region_name', 'Subgroup_category', 'Indicator_category'], axis=1)

	#get the different indicators as separate columns
	all_indicators = np.sort(df_behaviour['Indicator'].unique())
	df_behaviour_incols = pd.get_dummies(df_behaviour['Indicator']).reindex(columns = all_indicators, fill_value = 0)

	df_behaviour_incols.update(df_behaviour_incols.mul(df_behaviour.Value, 0))

	#then group by
	df_behaviour_incols = df_behaviour_incols.groupby(['Date'])[all_indicators].sum().reset_index().sort_index()#.sort_values('Date')
	df_behaviour_incols.set_index('Date', inplace = True)

	#replace unreasonably low values with nans
	df_behaviour_incols[df_behaviour_incols < 2] = np.nan

	#interpolate
	#resample to once a day
	df_behaviour_incols = df_behaviour_incols.resample('1d').mean()

	#then interpolate
	df_behaviour_incols = df_behaviour_incols.interpolate(method = 'linear')

	#pre-extrapolate columns with nans
	for col in df_behaviour_incols.columns:
		if np.sum(df_behaviour_incols[col].isna()) > 0:
			#interpolate this one columns
			df_sub = extrapolate_dataframe(df_behaviour_incols.loc[df_behaviour_incols[col].notna()][[col]], col, df_behaviour_incols.index[0], base_period = 7)

			#insert back into dataframe
			df_behaviour_incols[col] = df_sub[col]

	### now we need to perform imputation, filling in the blanks before the
	### survey starts
	#extrapolate at the start
	if startdate < np.datetime64(df_behaviour_incols.index[0]):
		df_sub = pd.DataFrame()
		for col in df_behaviour_incols.columns:
			df_sub[col] = extrapolate_dataframe(df_behaviour_incols[[col]], col, startdate, base_period = 7)[col]

		df_behaviour_incols = df_sub

	#extrapolate at the end
	if enddate > np.datetime64(df_behaviour_incols.index[0]):
		df_sub = pd.DataFrame()
		for col in df_behaviour_incols.columns:
			df_sub[col] = extrapolate_dataframe(df_behaviour_incols[[col]], col, enddate, base_period = 7)[col]

		df_behaviour_incols = df_sub

	return df_behaviour_incols

def load_mortality_hosp_agegroup():
	"""
	Load data on mortality and hospitalization per age group
	"""
	### First load mortality data
	df_individual_agegroups = load_individual_positive_test_data(load_agegroups = True, load_datatype = 'deaths', correct_for_pop_pyramid = False)

	#sum accross all dates
	df_deaths_agegroups = pd.DataFrame(df_individual_agegroups.sum(axis = 0), columns = ['N_deceased'])
	#convert to percentages
	totaldeaths = df_deaths_agegroups['N_deceased'].sum()
	df_deaths_agegroups['Deceased_fraction'] = df_deaths_agegroups['N_deceased']/totaldeaths

	### now load hospitalization data
	df_individual_agegroups = load_individual_positive_test_data(load_agegroups = True, load_datatype = 'hosp', correct_for_pop_pyramid = False)

	#sum accross all dates
	df_hosp_agegroups = pd.DataFrame(df_individual_agegroups.sum(axis = 0), columns = ['N_hosp'])
	#convert to percentages
	totalhosp = df_hosp_agegroups['N_hosp'].sum()
	df_hosp_agegroups['Hosp_fraction'] = df_hosp_agegroups['N_hosp']/totalhosp

	### determine mortality percentages for groups <50 based on literature IFR
	ifr_dict = {
				'0-9': 0.00004,
				'10-19': 0.00004,
				'20-29': 0.00021,
				'30-39': 0.0009,
				'40-49': 0.002
				}
	df_IFR = pd.DataFrame.from_dict({'Age': list(ifr_dict.keys()), 'IFR': list(ifr_dict.values())})
	df_IFR.set_index('Age', inplace = True)

	#determine fraction of deaths that can be attributed to age groups <50
	df_IFR['Fraction'] = df_IFR['IFR']/df_IFR['IFR'].sum()

	for agegroup in df_IFR.index:
		df_deaths_agegroups.loc[agegroup] = df_deaths_agegroups.loc['<50'] * df_IFR.loc[agegroup]['Fraction']
	df_deaths_agegroups = df_deaths_agegroups.drop('<50')
	df_deaths_agegroups = df_deaths_agegroups.sort_index()

	#apply the same estimation to hospitalization, so also based on IFR
	#note that some of the persons hospitalized with age <50 are already noted
	for agegroup in df_IFR.index:
		df_hosp_agegroups.loc[agegroup] += df_hosp_agegroups.loc['<50'] * df_IFR.loc[agegroup]['Fraction']
	df_hosp_agegroups = df_hosp_agegroups.drop('<50')
	df_hosp_agegroups = df_hosp_agegroups.sort_index()

	#now join the dataframes
	df_deaths_hosp_agegroups = df_hosp_agegroups.join(df_deaths_agegroups, how = 'inner')

	return df_deaths_hosp_agegroups


def plot_prevalence_R():
	df_prevalence, df_Rt = load_prevalence_Rt_data()
	df_response = load_government_response_data()

	#filter on starting date
	startdate = '2020-02-15'
	df_prevalence = df_prevalence.loc[df_prevalence.index > startdate]
	df_Rt = df_Rt.loc[df_Rt.index > startdate]
	df_response = df_response.loc[df_response.index > startdate]

	print('----\nLatest data')
	print(f'Prevalence: {df_prevalence.loc[df_prevalence["prev_avg"].notnull()].index[-1]}')
	print(f'R: {df_Rt.loc[df_Rt["Rt_avg"].notnull()].index[-1]}')
	print('----')

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax3 = ax1.twinx()
	ax3.spines['right'].set_position(('axes', 1.12))

	#plot prevalence
	ln1 = ax1.plot(df_prevalence.index, df_prevalence['prev_avg'], label = 'Prevalence')
	ax1.fill_between(df_prevalence.index, df_prevalence['prev_low'], df_prevalence['prev_up'], alpha = 0.4)

	#plot R
	ln2 = ax2.plot(df_Rt.index, df_Rt['Rt_avg'], color = 'maroon', label = r'$R$')
	ax2.fill_between(df_Rt.index, df_Rt['Rt_low'], df_Rt['Rt_up'], alpha = 0.4, color = 'maroon')

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

	plt.savefig(f'{R_plotloc}Mobility_change_Google.png', dpi = 200, bbox_inches = 'tight')
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

	plt.savefig(f'{R_plotloc}Mobility_change_Apple.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_sewage():
	"""
	Plot measurements of SARS-CoV-2 RNA per ml measurements in sewage
	"""
	df_sewage = load_sewage_data(smooth = True, shiftdates = True)

	df_sewage = df_sewage.loc[df_sewage.index > '2020-08-05']

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ln1 = ax1.plot(df_sewage.index, df_sewage.n_measurements, color = '#0C83CC', label = 'Number of measurements', alpha = 1)

	ax2.scatter(df_sewage.index, df_sewage['RNA_flow_per_100000'], color = betterorange, label = 'Average RNA abundance', alpha = 0.6, s = 5)
	ln2 = ax2.plot(df_sewage.index, df_sewage['RNA_flow_smooth'], color = betterorange, label = 'Average RNA abundance smoothed')

	indicate_school_closed(ax2)

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

def plot_daily_results(use_individual_data = True, startdate = '2020-07-01'):
	"""
	Plot up to date test results
	"""
	if use_individual_data:
		df_daily_covid = load_individual_positive_test_data(load_agegroups = False)

		df_daily_covid = df_daily_covid.rename(columns = {'N_cases': 'Total_reported'})
	else:
		df_daily_covid = g(correct_for_delay = False)
	df_n_tests = load_number_of_tests(enddate = df_daily_covid.index.values[-1], ignore_last_datapoint = False)

	df_response = load_government_response_data()

	### correct for the delay between the onset of symptoms and the result of the test
	#for source of incubation period, see the readme
	incubation_period = 6 #days, left of average of 8.3 due to extremely skewed distribution
	#test delay determined from anecdotal evidence
	time_to_test_delay = 3 #days
	result_delay = 1

	if not use_individual_data:
		#correct the results to the day of the test
		df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{result_delay} day')

		#merge datasets
		df_daily_covid = df_daily_covid.merge(df_n_tests[['Number_of_tests', 'Extrapolated']], right_index = True, left_index = True)

		#correct the results to the day of infection
		df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{int(incubation_period + time_to_test_delay)} day')
	else:
		#shift number of tests to date of disease onset (DDO)
		df_n_tests.index = df_n_tests.index - pd.Timedelta(f'{int(time_to_test_delay)} day')

		#merge datasets
		df_daily_covid = df_daily_covid.merge(df_n_tests[['Number_of_tests', 'Extrapolated']], right_index = True, left_index = True)

		#correct the results to the day of infection
		df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{int(incubation_period)} day')

	#determine test positivity rate
	df_daily_covid['Positivity_ratio'] = df_daily_covid['Total_reported']/df_daily_covid['Number_of_tests']


	#select second wave of infections
	df_daily_covid = df_daily_covid.loc[df_daily_covid.index > startdate]
	df_response = df_response.loc[df_response.index > startdate]


	### make the plot
	fig, ax1 = plt.subplots(figsize = (5, 4))

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

	if use_individual_data:
		lns5 += indicate_incomplete_test_data(ax1, df_daily_covid)

	lns = lns1 + lns2 + lns3 + lns4 + lns5
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc = 'lower left', prop = {'size': 6}) #loc = 'center', bbox_to_anchor = (0.5, -0.4), ncol = 2,
	# fig.subplots_adjust(bottom = 0.2, right = 0.8)

	if use_individual_data:
		ax1.set_xlabel(f'Estimated infection date\n(date of disease onset - incubation period (~6 days))')
	else:
		ax1.set_xlabel(f'Estimated infection date\n(reporting date - incubation period (~6 days) - test delays (~{time_to_test_delay + result_delay} days))')
	ax1.set_ylabel('Number of tests per day')
	ax2.set_ylabel('Positivity rate [%]')
	ax3.set_ylabel('Oxford Stringency Index')

	ax1.set_title(f'SARS-CoV-2 tests in the Netherlands up to {str(df_daily_covid.index[-1].date())}')

	ax1.xaxis.set_tick_params(rotation = 20)
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	# fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax1.xaxis.set_major_formatter(myfmt)

	ax1.set_ylim(0)
	ax2.set_ylim(0)
	ax3.set_ylim(0)

	if use_individual_data:
		savename = f'{plotloc}Tests_second_wave_individual_reports.png'
	else:
		savename = f'{plotloc}Tests_second_wave_daily_reported.png'

	plt.savefig(savename, dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_hospitalization():
	"""

	"""
	df_IC = load_IC_data()
	df_prevalence, df_Rt = load_prevalence_Rt_data()

	### determine fraction of infected people on the IC
	#first go back to absolute number of people infectious
	df_prevalence['prev_avg'] /= per_million_factor

	df_IC_rate = df_prevalence.merge(df_IC, right_index = True, left_index = True)
	df_IC_rate['IC_rate'] = df_IC_rate['Amount']/df_IC_rate['prev_avg']
	#filter out nans
	df_IC_rate = df_IC_rate.loc[df_IC_rate['IC_rate'].notna()]

	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()

	lns1 = ax1.plot(df_IC.index, df_IC.Amount, label = 'Number of patients on IC')
	lns2 = ax2.plot(df_IC_rate.index, df_IC_rate.IC_rate, label = 'Fraction of infectious patients on IC', color = 'orangered')

	ax1.set_ylabel('Number of patients')
	ax1.set_title('IC occupation')

	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc='best')
	ax1.grid(linestyle = ':')

	fig.autofmt_xdate()

	plt.savefig(f'{plotloc}Hospital_IC_occupation.png', dpi = 200, bbox_inches = 'tight')
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

def plot_individual_data(use_agegroups = True):
	"""
	Plot data from individual cases, which indicate the date of disease onset
	(if known) and age
	"""
	df_individual = load_individual_positive_test_data(load_agegroups = use_agegroups)

	#correct the overall results to the day of infection
	df_individual.index = df_individual.index - pd.Timedelta(f'{int(incubation_period)} day')

	#select recent data
	df_individual = df_individual.loc[df_individual.index >= '2020-08-01']

	if use_agegroups:
		### Plot as a heatmap
		if True:

			#convert to an image
			imgdata = np.array(df_individual).T

			#vertical labels
			age_labels = list(df_individual.columns)

			#horizontal label dates
			x_labeldates = pd.date_range(np.min(df_individual.index), np.max(df_individual.index), freq = '14D', format = '%d-%m-%Y').to_series()

			fig, ax = plt.subplots(figsize = (8, 4))

			im = ax.imshow(imgdata*100, cmap = 'Reds', origin = 'lower')

			cbar = plt.colorbar(im)
			cbar.ax.set_ylabel('Percentage of total population')
			cbar.ax.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))

			ax.set_aspect(aspect = 6)

			ax.set_yticks(np.arange(10))
			ax.set_yticklabels(age_labels)
			ax.set_xticks(np.arange(0, imgdata.shape[1], 14))
			ax.set_xticklabels(list(x_labeldates.dt.strftime('%d-%m-%Y')))

			ax.set_xlabel('Estimated date of infection (date of disease onset - incubation period)')
			ax.set_ylabel('Age group')

			ax.set_title('Percentage of people per age group tested positive per day')

			# ax.xaxis.set_tick_params(rotation = 45)
			fig.autofmt_xdate()

			plt.savefig(f'{plotloc}Individual_testing_data_agegroups_heatmap.png', dpi = 200, bbox_inches = 'tight')
			plt.close()

		### plot age groups as a line graph
		if True:
			smoothscale = 3

			print(df_individual.tail(10))

			fig, ax = plt.subplots()

			cmap = plt.get_cmap('plasma')
			cNorm  = mcolors.Normalize(vmin = 0, vmax = len(df_individual.columns) - 1)
			scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

			for i, col in enumerate(df_individual.columns):
				#smooth the data
				ax.plot(df_individual.index, df_individual[col].rolling(smoothscale).mean(), label = col, color = scalarMap.to_rgba(i))

			ax.legend(loc = 'best', title = 'Age group')

			ax.grid(linestyle = ':')

			indicate_school_closed(ax)

			#indicate period where not yet everyone who will be tested, has been
			indicate_incomplete_test_data(ax, df_individual)

			#fix the date labels
			ax.set_xticks(pd.date_range(np.min(df_individual.index), np.max(df_individual.index), freq = '14D', format = '%d-%m-%Y').to_series())
			fig.autofmt_xdate()
			myfmt = mdates.DateFormatter('%d-%m-%Y')
			ax.xaxis.set_major_formatter(myfmt)

			ax.set_xlabel('Estimated date of infection (date of disease onset - incubation period)')
			ax.set_ylabel('Percentage of people in age group')
			ax.set_title('Percentage of people per age group tested positive each day')


			plt.savefig(f'{plotloc}Individual_testing_data_agegroups_line.png', dpi = 200, bbox_inches = 'tight')
			plt.close()
	else:
		### plot simply the number of positive tests per DDO
		fig, ax = plt.subplots()

		ax.plot(df_individual.index, df_individual.N_cases)

		ax.grid(linestyle = ':')

		ax.set_ylabel('Number of positive tests')
		ax.set_title('Number of positive tests on date of disease onset')

		fig.autofmt_xdate()
		myfmt = mdates.DateFormatter('%d-%m-%Y')
		ax.xaxis.set_major_formatter(myfmt)

		plt.savefig(f'{plotloc}Individual_testing_data_total_cases.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

def plot_cluster_change():
	"""
	Plot the change in COVID-19 cluster types
	"""

	df_clusters = load_cluster_data()

	df_clusters = df_clusters.loc[df_clusters.index > '2020-07-01']


	fig, ax = plt.subplots()


	plot_columns = {
		'School (basis) ': 'Elementary school',
		'School/Studie (overig) ': 'Highschool/college',
		'(Sport)vereniging ': '(Sports) association',
		'Bedrijf (overig) / Organisatie ':  'Company (other)',
		'Evenement / Feest ': 'Event/party',
		'Familie / Vrienden ': 'Family/friends',
		'Horeca / Vakantieverblijf ': 'Horeca/vacation stay',
		'Woonvoorziening / Verzorging ': 'Living arrangement/carehome'
	}

	cmap = plt.get_cmap('plasma')
	cNorm  = mcolors.Normalize(vmin = 0, vmax = len(plot_columns) - 1)
	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

	for i, key in enumerate(plot_columns.keys()):
		ax.plot(df_clusters.index, df_clusters[key], label = plot_columns[key], color = scalarMap.to_rgba(i))

	#also indicate school closures etc
	indicate_school_closed(ax)

	ax.set_ylabel('Number of clusters')
	ax.set_title('Number of COVID-19 clusters in different settings per week')

	ax.grid(linestyle = ':')
	ax.legend(loc = 'best', prop = {'size': 6.5})

	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax.xaxis.set_major_formatter(myfmt)

	plt.savefig(f'{plotloc}Cluster_amount_per_setting.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_R_versus_weather(startdate = '2020-06-15'):
	df_prevalence, df_R = load_prevalence_Rt_data()
	df_weather = load_weather_data(smooth = False, abs_hum = True)

	#merge datasets
	df_plot = df_R.join(df_weather, how = 'inner')

	df_plot = df_plot.loc[df_plot.index > startdate]

	fig, axs = plt.subplots(ncols = 3, sharey = True, figsize = (8.5, 4))
	axs = axs.flatten()

	colours = ['maroon', 'navy', 'green']
	params = ['TAvg', 'Rad', 'HumAbsAvg']
	labels = ['Daily average temperature [C]', 'Daily average solar radiation [J/cm$^2$]', 'Daily average absolute humidity [g/kg]']

	hum_threshold = 8 #g/kg
	print('-----------')
	print(f'R on days with absolute humidity below threshold of {hum_threshold} g/kg:')
	print(np.mean(df_plot.loc[df_plot['HumAbsAvg'] < hum_threshold]['Rt_avg']))
	print(f'\nR on days with absolute humidity above threshold of {hum_threshold} g/kg:')
	print(np.mean(df_plot.loc[df_plot['HumAbsAvg'] > hum_threshold]['Rt_avg']))
	print('-----------')

	for i in range(len(params)):
		axs[i].scatter(df_plot[params[i]], df_plot['Rt_avg'], facecolor = colours[i], edgecolor = 'none', s = 3)

		axs[i].grid(linestyle = ':')
		axs[i].set_xlabel(labels[i], fontsize = 8.5)

		# axs[i].xaxis.set_major_locator(MaxNLocator(prune = 'upper'))

		if i == 0:
			axs[i].set_ylabel('R')

	fig.subplots_adjust(wspace = 0)

	axs[1].set_title(f'R versus various weather observables since {startdate}')

	plt.savefig(f'{R_plotloc}R_dependence_weather.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def plot_longterm_prevalence_decay(given_R = 0.9, enddate = '2021-05-01'):
	"""
	Plot the decay of the prevalence at a given R regime in the future
	"""
	df_prevalence, df_Rt = load_prevalence_Rt_data()
	# df_response = load_government_response_data()

	#filter on starting date
	startdate = '2020-03-01'
	df_prevalence = df_prevalence.loc[df_prevalence.index > startdate]
	df_Rt = df_Rt.loc[df_Rt.index > startdate]
	# df_response = df_response.loc[df_response.index > startdate]

	#find last date with known prevalence
	last_prev_date = df_prevalence.loc[df_prevalence['prev_avg'].notna()].index[-1]
	#find last date with known R
	last_R_date = df_Rt.loc[df_Rt['Rt_avg'].notna()].index[-1]

	enddate = pd.to_datetime(enddate, format = '%Y-%m-%d')

	timestep_size = 1

	#create rows to desired date
	while df_prevalence.index[-1] < enddate:
		df_prevalence.loc[df_prevalence.index[-1] + pd.Timedelta(f'{timestep_size} day')] = [np.nan]*3
	while df_Rt.index[-1] < enddate:
		df_Rt.loc[df_Rt.index[-1] + pd.Timedelta(f'{timestep_size} day')] = [np.nan]*3

	#fill last rows of R dataframe with given R
	df_Rt = df_Rt.set_value(df_Rt.index[df_Rt.index > last_R_date], 'Rt_avg', given_R)

	#determine which rows need filling
	fillmask = df_prevalence.index > last_prev_date
	already_known_mask = df_prevalence.index <= last_prev_date
	#determine number of days since end of known prevalence
	n_days = np.array((df_prevalence.loc[fillmask].index - last_prev_date).days)

	#predict coming decay based on R and current decay
	predicted_prev = exponential_model(df_prevalence.loc[last_prev_date]['prev_avg'], given_R, n_days, serial_interval)
	df_prevalence.loc[fillmask,'prev_avg'] = predicted_prev

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	#plot known prevalence
	ln1 = ax1.plot(df_prevalence.loc[already_known_mask].index, df_prevalence.loc[already_known_mask]['prev_avg'], label = 'Prevalence (known)', color = mplblue)
	ax1.fill_between(df_prevalence.loc[already_known_mask].index, df_prevalence.loc[already_known_mask]['prev_low'], df_prevalence.loc[already_known_mask]['prev_up'], alpha = 0.4, color = mplblue)

	#plot predicted prevalence
	ln2 = ax1.plot(df_prevalence.loc[fillmask].index, df_prevalence.loc[fillmask]['prev_avg'], label = 'Prevalence (predicted)', linestyle = '--', color = mplblue)

	#plot known R
	known_R_mask = df_Rt.index <= last_R_date
	ln3 = ax2.plot(df_Rt.loc[known_R_mask].index, df_Rt.loc[known_R_mask]['Rt_avg'], color = 'maroon', label = r'$R$ (known)')
	ax2.fill_between(df_Rt.loc[known_R_mask].index, df_Rt.loc[known_R_mask]['Rt_low'], df_Rt.loc[known_R_mask]['Rt_up'], alpha = 0.4, color = 'maroon')

	#plot given R
	ln4 = ax2.plot(df_Rt.loc[df_Rt.index > last_R_date].index, df_Rt.loc[df_Rt.index > last_R_date]['Rt_avg'], color = 'maroon', label = r'$R$ (set)', linestyle = '--')

	ax1.set_ylim(0)
	ax2.set_ylim(0, 2.6)

	lns = ln1 + ln2 + ln3 + ln4
	labs = [l.get_label() for l in lns]
	ax2.legend(lns, labs, loc = 'best')

	ax1.grid(linestyle = ':', axis = 'x')
	ax2.grid(linestyle = ':', axis = 'y')

	ax1.xaxis.set_tick_params(rotation = 45)

	# fig.autofmt_xdate()

	ax1.set_ylabel('Prevalence (estimated active cases per million)')
	ax2.set_ylabel(r'Reproductive number $R$')

	ax1.set_title('COVID-19 statistics of the Netherlands')

	plt.savefig(f'{plotloc}longterm_prevalence_decay_R={given_R}.png', dpi = 200, bbox_inches = 'tight')


def government_response_results_simple():
	"""
	Make a simple model of the effects of different responses of a government to
	the coronavirus pandemic
	"""

	def response_model_1(prevalence, prevalence_threshold, R_from_now, t_from_now, response_delay = 14, upward_R = 1.3, downward_R = 0.8):
		"""
		Model that ensures that a Rt below 1 is enforced when the prevalence rises
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


	df_prevalence, df_Rt = load_prevalence_Rt_data()

	response_delay = 14

	### tweakables
	starting_prev = 200 #per million
	prevalence_threshold = 2500 #per million

	upward_R = 1.2
	downward_R = 0.7

	#time range in days
	timestep_size = 1
	t_range = np.arange(0, 280, timestep_size)

	#store prevalence and Rt
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

def epidemiological_modelling(startdate = '2021-01-20'):
	"""
	Model the evolution of the coronavirus pandemic using a SEISD model. More info:
	https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model

	Also look in the readme.
	"""
	def get_mean_mu(df_prevalence, df_daily_covid, enddate = '2021-01-20'):
		startdate_plot = '2020-02-27'
		startdate_determining_avg = '2020-10-01'

		#slice the dD/dt between dates
		mask = (df_daily_covid.index > startdate_plot) & (df_daily_covid.index <= enddate)
		df_deaths = df_daily_covid.copy().loc[mask]

		#do the same for the prevalence I
		mask = (df_prevalence.index > startdate_plot) & (df_prevalence.index <= enddate)
		df_prevalence = df_prevalence.loc[mask]

		df_mu = df_daily_covid[['Deceased']].merge(df_prevalence[['prev_avg']], right_index = True, left_index = True)

		#now calculate the mortality rate per unit time
		df_mu['mu'] = df_mu['Deceased']/df_mu['prev_avg'] #day^-1

		fig, ax = plt.subplots()

		ax.plot(df_mu.index, df_mu['mu'])

		ax.set_ylabel(r'Mortality rate per unit time ($\mu$) [day$^-1$]')
		ax.set_title('Change in COVID-19 mortality rate per unit time ')

		ax.grid(linestyle = ':')

		fig.autofmt_xdate()

		plt.savefig(f'{epidem_modelling_plotloc}mu_change_NL.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

		#determine average between april and june
		mask = (df_mu.index > startdate_determining_avg) & (df_mu.index <= enddate)
		mu_mean = np.mean(df_mu.loc[mask]["mu"])
		print(f'Mean mu: {mu_mean}')

		#between 2020-10-01 and 2021-01-20 mu is on average 0.00862225
		#-> 1/mu ~ 116 days, which is a bit higher than China at their end of the pandemic.

		return mu_mean

	def beta_from_Rt(Rt, gamma, mu):
		return Rt * (gamma + mu)

	def array_relu(x):
		return np.max([np.zeros(len(x)), x], axis = 0)

	def IC_removal(param, pop):
		"""
		If the total number of persons (across age groups) exceeds the IC capacity,
		then first start removing elderly from the IC
		"""

		if np.sum(pop['IC']) - param['Mic'] > 0:
			#iteratively remove people from IC
			remove_IC = np.zeros(len(pop['IC']))

			#determine how many people still need to be removed from the IC
			n_need_to_be_removed = np.sum(pop['IC'] - remove_IC) - param['Mic']
			i = len(remove_IC) - 1

			#iterate through the age groups from old to young until enough people are
			#removed from the IC
			while n_need_to_be_removed > 0:
				if n_need_to_be_removed > pop['IC'][i]:
					remove_IC[i] = pop['IC'][i]
				else:
					remove_IC[i] = n_need_to_be_removed

				i -= 1
				n_need_to_be_removed = np.sum(pop['IC'] - remove_IC) - param['Mic']

			return remove_IC
		else:
			return np.zeros(len(pop['IC']))

	### differential equations of the model
	def dSdt(param, pop):
		return -param['beta'] * pop['S'] * pop['I']/pop['N'] + param['rho'] * pop['R']

	def dEdt(param, pop):
		return param['beta'] * pop['S'] * pop['I']/pop['N'] - param['a']*pop['E']

	def dIdt(param, pop):
		return (1 - param['i']) * param['a'] * pop['E'] - (param['gamma'] + param['mu']) * pop['I']

	def dICdt(param, pop):
		#IC overcapacity term
		return param['i'] * param['a'] * pop['E'] - (param['gamma'] + param['mu']) * pop['IC'] - IC_removal(param, pop)

	def dRdt(param, pop):
		return param['gamma'] * (pop['I'] + pop['IC']) - param['rho'] * pop['R']

	def dDdt(param, pop):
		#IC overcapacity term
		return param['mu'] * (pop['I'] + pop['IC']) + IC_removal(param, pop)

	def column_to_2D_array(series):
		"""
		Convert a pandas dataframe column with each entry a constant length
		numpy array to a 2D numpy array
		"""
		columnlength = len(series)
		array_in_row_length = len(series.iloc[0])

		X = np.zeros((columnlength, array_in_row_length))

		for i, row in enumerate(series.values):
			X[i] = row

		return X

	def plot_epidemiological_model(df_pop, IC_frac_of_I, IC_limit, df_pop_pyramid):
		params_to_plot = ['S', 'E', 'I', 'R']

		all_columns = params_to_plot + ['IC', 'D']

		df_pop.set_index('Days', inplace = True)

		pop_result_dict = {}
		for col in all_columns:
			pop_result_dict[col] = column_to_2D_array(df_pop[col])

		#Plot the total SEIRD+IC values
		if True:
			fig, ax1 = plt.subplots()

			ax2 = ax1.twinx()
			ax3 = ax1.twinx()

			ax3.spines['right'].set_position(('axes', 1.14))

			colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

			#we can estimate the number of people on the IC based on the fraction
			#of infectious people that end up on the IC, see plot_hospitalization
			lns = ax2.plot(df_pop.index, np.sum(pop_result_dict['IC'], axis = 1)/1e3, label = 'IC', color = 'black')
			lns += ax3.plot(df_pop.index, np.sum(pop_result_dict['D'], axis = 1)/1e3, label = 'D', color = '#d62728')

			for i, p in enumerate(params_to_plot):
				lns += ax1.plot(df_pop.index, np.sum(pop_result_dict[p], axis = 1)/1e6, label = p, color = colours[i])

			#also show IC limit
			xlims = ax2.get_xlim()
			ax2.hlines(IC_limit/1e3, xlims[0], xlims[1], color = 'dimgray', linestyle = '--')
			ax2.set_xlim(xlims)

			ax1.grid(linestyle = ':')

			ax1.set_xlabel('Days since start')
			ax1.set_ylabel(r'Number of people [$\times 10^6$]')
			ax2.set_ylabel(r'Number of people in IC [$\times 10^3$]')
			ax3.set_ylabel(r'Number of deceased [$\times 10^3$]')

			ax1.set_title('SEIRSD+IC epidemiological model of COVID-19 in the Netherlands\nstarting at 2021-01-20')

			ax1.set_ylim(0)
			ax2.set_ylim(0)
			ax3.set_ylim(0)

			labs = [l.get_label() for l in lns]
			ax1.legend(lns, labs, loc='best')
			ax1.grid(linestyle = ':')

			plt.savefig(f'{epidem_modelling_plotloc}Model_result_population.png', dpi = 200, bbox_inches = 'tight')
			plt.close()

		#plot only deaths per age group
		if True:
			fig, ax1 = plt.subplots()

			ax2 = ax1.twinx()

			# cmap = plt.get_cmap('Oranges')
			cmap = turbo_cmap
			cNorm  = mcolors.Normalize(vmin = 0, vmax = len(df_pop_pyramid) - 1)
			scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

			lns = ax1.plot(df_pop.index, np.sum(pop_result_dict['IC'], axis = 1)/1e3, label = 'IC', color = 'black', linestyle = '-.')

			for i, agegroup in enumerate(df_pop_pyramid.index):
				lns += ax2.plot(df_pop.index, pop_result_dict['D'][:,i]/df_pop_pyramid.loc[agegroup].N_people, label = agegroup, color = scalarMap.to_rgba(i))

			#also show IC limit
			xlims = ax1.get_xlim()
			ax1.hlines(IC_limit/1e3, xlims[0], xlims[1], color = 'dimgray', linestyle = '--')
			ax1.set_xlim(xlims)

			ax1.grid(linestyle = ':')

			ax1.set_xlabel('Days since start')
			ax1.set_ylabel(r'Number of persons in IC [$\times 10^3$]')
			ax2.set_ylabel(r'Fraction of age group deceased')

			ax1.set_title('SEIRSD+IC epidemiological model of COVID-19 in the Netherlands\nstarting at 2021-01-20')

			ax1.set_ylim(0)
			ax2.set_ylim(0)

			labs = [l.get_label() for l in lns]
			ax1.legend(lns, labs, loc = 'upper left', prop = {'size': 8})
			ax1.grid(linestyle = ':')

			plt.savefig(f'{epidem_modelling_plotloc}Model_result_deceased.png', dpi = 200, bbox_inches = 'tight')
			plt.close()

	df_prevalence, df_Rt = load_prevalence_Rt_data()
	#go back to absolute numbers right away
	df_prevalence['prev_avg'] /= per_million_factor
	df_IC = load_IC_data()
	df_daily_covid = load_daily_covid(correct_for_delay = False)
	df_individual_postest = load_individual_positive_test_data(load_agegroups = True, correct_for_pop_pyramid = False)
	df_pop_pyramid = load_pop_pyramid()
	df_mortality_hosp_agegroup = load_mortality_hosp_agegroup()

	# get_mean_mu(df_prevalence, df_daily_covid)

	agegroups = df_mortality_hosp_agegroup.index.values

	#compare hospitalization fractions of agegroups with CDC data, compares reasonably
	#https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html
	# print(df_mortality_hosp_agegroup['Hosp_fraction']/df_mortality_hosp_agegroup.loc['20-29']['Hosp_fraction'])

	### determine average distribution of infections since the summer across age groups
	#do not yet use the population pyramid corrected data, this will be done later
	df_individual_postest = df_individual_postest.loc[(df_individual_postest.index > '2020-07-01') & (df_individual_postest.index < '2021-01-01')]
	#sum across the dates
	df_individual_postest = pd.DataFrame(df_individual_postest.sum(axis = 0), columns = ['Age'])
	#normalize
	df_individual_postest['Age_fraction'] = df_individual_postest['Age'] / df_individual_postest['Age'].sum()

	### determine recovery rate fractions across age groups
	### this does not sum to 1 across age groups, it is the fraction per age group
	### that recovers
	df_mortality_hosp_agegroup['Recovery_fraction'] = (df_individual_postest.Age.values - df_mortality_hosp_agegroup.N_deceased)/df_individual_postest.Age.values

	#time step size in days
	dt = 1
	N_days = 250
	N_steps = int(N_days/dt)

	IC_frac_of_I = 0.005
	#based on https://covid-analytics.nl/index.html
	IC_limit = 1900

	# N = 17407585 #total population of the Netherlands on 1-1-2020
	N = df_pop_pyramid.N_people.values
	N_tot = np.sum(N)

	### First set several rate parameters
	gamma = 1/13.15
	mu_0 = 0.00049 #based on Dutch data, very similar to data from China
	#spread across age groups using mortality rates, correcting for the
	#population pyramid
	mu = mu_0 * df_mortality_hosp_agegroup.Deceased_fraction.values/np.mean(df_mortality_hosp_agegroup.Deceased_fraction) / (N/np.mean(N))
	Rt = 1.5
	beta = beta_from_Rt(Rt, gamma, mu)

	param = {
	'beta': beta,
	'a': 1/2.5, #person is about 2.5 days not contagious
	'gamma': gamma,
	'mu': mu,
	'rho': 1/(6*30), #immunity for ~6 months
	'i': IC_frac_of_I * (df_mortality_hosp_agegroup.Hosp_fraction.values/np.mean(df_mortality_hosp_agegroup.Hosp_fraction)) / (N/np.mean(N)),
	'Mic': IC_limit
	}

	### initialize our population at t = 0
	#distribute prevalence across the age groups based on positive test results,
	#correcting for the size of the age groups
	I = df_prevalence.loc[startdate]['prev_avg'] * df_individual_postest.Age_fraction.values
	#distribute IC occupancy based on known hospitality fractions
	IC = df_IC.loc[startdate]['Amount'] * df_mortality_hosp_agegroup.Hosp_fraction.values
	#estimate exposed people based on recovery rate gamma and exposure period 1/a
	E = I*(param['gamma'] + param['mu']) / param['a']
	D = np.zeros(len(agegroups))
	#estimate number of immune people by summing the prevalence in the immunity period
	#times the data timestep (1 day, so omitted) and dividing by the average time to recovery
	R = df_prevalence.loc[
			(df_prevalence.index > (np.datetime64(startdate) - np.timedelta64(int(1/param['rho']), 'D'))) &
			(df_prevalence.index < (np.datetime64(startdate) - np.timedelta64(int(1/param['gamma']), 'D')))]['prev_avg'].sum() / (1/param['gamma'])
	#now distribute across the age groups based on positive test results
	#where deceased are removed from the data
	R = R * (N/N_tot) * df_mortality_hosp_agegroup.Recovery_fraction.values

	#get susceptible population
	S = N - I - IC - E - R - D

	pop = {
	'Days': 0,
	'N': N,
	'S': S,
	'E': E,
	'I': I,
	'IC': IC,
	'R': R, #start counting deaths from now on
	'D': D #start counting deaths from now on
	}

	#dataframe which will keep track of changes in population
	df_pop = pd.DataFrame(columns = list(pop.keys()))

	### naive integrator
	for i in tqdm(range(N_steps)):
		#determine change
		dS = dt * dSdt(param, pop)
		dE = dt * dEdt(param, pop)
		dI = dt * dIdt(param, pop)
		dIC = dt * dICdt(param, pop)
		dR = dt * dRdt(param, pop)
		dD = dt * dDdt(param, pop)

		#apply change
		pop['Days'] += dt
		pop['S'] = pop['S'] + dS
		pop['E'] = pop['E'] + dE
		pop['I'] = pop['I'] + dI
		pop['IC'] = pop['IC'] + dIC
		pop['R'] = pop['R'] + dR
		pop['D'] = pop['D'] + dD

		df_pop = df_pop.append(pop, ignore_index = True)

	plot_epidemiological_model(df_pop, IC_frac_of_I, IC_limit, df_pop_pyramid)



def stringency_R_correlation(enddate = '2020-11-26'):
	df_response = load_government_response_data()
	df_prevalence, df_Rt = load_prevalence_Rt_data()

	#merge datasets
	df_reponse_results = df_response.merge(df_Rt[['Rt_avg']], right_index = True, left_index = True)

	#select date range
	mask = (df_reponse_results.index > '2020-02-16') & (df_reponse_results.index <= enddate)
	df_reponse_results = df_reponse_results.loc[mask]

	### fit a linear model
	#select only high stringency to reflect the current day situation
	high_stringency_mask = df_reponse_results['StringencyIndex'] > 30
	popt, perr, r_squared = fit_model(linear_model,
						df_reponse_results.loc[high_stringency_mask]['StringencyIndex'],
						df_reponse_results.loc[high_stringency_mask]['Rt_avg'])

	#get number of days since start as another column
	df_reponse_results['N_days'] = (df_reponse_results.index - df_reponse_results.index[0]).days

	### plot results
	fig, ax = plt.subplots()

	## we want to color the points by the age since the start of the pandemic
	cmap = plt.get_cmap('viridis')
	cNorm = mcolors.Normalize(vmin = 0, vmax = len(df_reponse_results['N_days']) - 1)
	scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

	scatterplot = ax.scatter(df_reponse_results['StringencyIndex'], df_reponse_results['Rt_avg'], facecolor = scalarMap.to_rgba(df_reponse_results['N_days']), edgecolor = 'none', alpha = 0.7, s = 10, label = 'RIVM inferred $R$')

	#plot the model
	xpoints = np.linspace(np.min(df_reponse_results.loc[high_stringency_mask]['StringencyIndex']), np.max(df_reponse_results.loc[high_stringency_mask]['StringencyIndex']), num = 500)
	ax.plot(xpoints, linear_model(xpoints, *popt), label = r'Fit ($R^2 = $' + f'{r_squared:0.03f})', color = 'black')

	### now plot the effects of a variant which is more infectious
	mean_R_increase = 1.43
	errorbar = 0.15

	xpoints = np.linspace(np.min(df_reponse_results.loc[high_stringency_mask]['StringencyIndex']), 100, num = 500)
	R_prediction_new_variant = linear_model(xpoints, *popt) * mean_R_increase

	ax.plot(xpoints, R_prediction_new_variant, label = f'SARS-COV-2 20B/501Y.V1 model\nwith R {mean_R_increase} times higher', color = betterorange)
	ax.fill_between(xpoints, R_prediction_new_variant - errorbar, R_prediction_new_variant + errorbar, color = betterorange, alpha = 0.4)


	ax.set_xlabel('Oxford Stringency Index')
	ax.set_ylabel(f'$R$')

	ax.grid(linestyle = ':', zorder = -10)
	ax.legend(loc = 'best', prop={'size': 9})

	ax.set_title(r'$R$ versus stringency index of Dutch coronavirus reponse')

	plt.savefig(f'{plotloc}Stringency_R_correlation.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


def estimate_recent_R(enddate_train = '2020-10-25', regression_method = 'ridge'):
	"""
	Estimate the reproductive number R with mobility data
	"""
	print(f'WARNING: end date for R training set is {enddate_train}')

	startdate_train = '2020-06-15'
	startdate_pred = '2020-07-01'

	df_prevalence, df_R = load_prevalence_Rt_data()
	df_google_mob, df_apple_mob = load_mobility_data(smooth = True)
	df_weather = load_weather_data(smooth = True, abs_hum = True)
	#for the plot as a reference as well as data input
	df_response = load_government_response_data()
	df_response_plot = df_response.loc[df_response.index > startdate_pred]

	print('------------')
	print('R prediction input data latest entries:')
	print('Google mob: ' + str(df_google_mob.index[-1].date()))
	print('Apple mob: ' + str(df_apple_mob.index[-1].date()))
	print('Weather data: ' + str(df_weather.index[-1].date()))
	print('Government response data: ' + str(df_response.index[-1].date()))
	print('------------')

	#determine error of R
	df_R['Rt_abs_error'] = ((df_R['Rt_low'] - df_R['Rt_avg']).abs() + (df_R['Rt_up'] - df_R['Rt_avg']).abs())/2

	#merge datasets
	# df_mob_R = df_google_mob.merge(df_R, right_index = True, left_index = True)
	#left join with df_R is essential otherwise we limit ourselves to the date
	#range of the RIVM estimations, which we actually want to surpass
	df_mob_R = df_google_mob.join(df_R, how = 'left')
	df_mob_R = df_mob_R.join(df_apple_mob, how = 'inner')
	df_mob_R = df_mob_R.join(df_response, how = 'inner')
	df_mob_R = df_mob_R.join(df_weather, how = 'inner')

	print('Combined data: ' + str(df_mob_R.index[-1].date()))
	print('------------')

	#only now load the behaviour data as we have to extrapolate this anyhow
	#and need to get the extrapolation edges
	# df_behaviour = load_behaviour_data(np.datetime64(startdate_train), np.datetime64(df_mob_R.index[-1]))
	# print(df_behaviour.columns)
	# #and also merge this
	# df_mob_R = df_mob_R.join(df_behaviour, how = 'inner')

	#select date range
	mask = (df_mob_R.index > startdate_train) & (df_mob_R.index <= enddate_train)
	df_train = df_mob_R.loc[mask]
	df_pred = df_mob_R.loc[df_mob_R.index > startdate_pred]


	key_names = {
	'retail_recreation_smooth': 'Retail & recreation',
	'parks_smooth': 'Parks',
	'transit_stations_smooth': 'Transit stations',
	'workplaces_smooth': 'Workplaces',
	'residential_smooth': 'Residential',
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

		plt.savefig(f'{R_plotloc}Mobility_R_correlation.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

	### determine correlation matrix for all these parameters
	if True:
		compare_parameters = [
			'Rt_avg',
			'retail_recreation_smooth',
			'transit_stations_smooth',
			'residential_smooth',
			'workplaces_smooth',
			'driving_smooth',
			'walking_smooth',
			'transit_smooth',
			'Rad',
			'TAvg',
			'HumAbsAvg'
		]

		#remove the "_smooth" part in the parameters names for better plotting
		compare_parameters_names = [param.replace('_smooth', '') for param in compare_parameters]

		#get the multiple parameters into a single array
		A_data = dataframes_to_NDarray(df_train, compare_parameters)

		'''
		### Remove multiple colinearity by applying a whitening transformations
		Xwhitened = whitening_transform(A_data[:,1:])
		#append data on R
		A_whitened = np.concatenate((A_data[:,:1], Xwhitened), axis = 1)

		#determine the correlation matrix
		corr_matrix = np.corrcoef(A_whitened.T)


		### plot the transformed data to see what they respresent
		#get correlation coefficients
		corr_coeff = corr_matrix[0]
		#select on large enough absolute correlation coefficient
		corr_coeff_sel = corr_coeff[np.abs(corr_coeff) > 0.15]

		sortloc = np.argsort(np.abs(corr_coeff_sel))

		fig, ax = plt.subplots()

		cmap = plt.get_cmap('viridis')
		cNorm  = mcolors.Normalize(vmin = 0, vmax = len(sortloc) - 1)
		scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

		xpoints = np.arange(A_whitened.shape[0])

		for i, loc in enumerate(sortloc):
			ax.plot(xpoints, A_whitened[:,loc], color = scalarMap.to_rgba(i), label = f'{corr_coeff_sel[loc]:0.03f}')

		ax.grid(linestyle = ':')
		ax.legend(loc = 'best', title = 'CorrCoeff')
		ax.set_xlabel('Time in days')
		ax.set_ylabel('Parameter value')

		plt.savefig(f'{R_plotloc}R_mob_data_whitened.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

		del fig, ax
		'''

		corr_matrix = np.corrcoef(A_data.T)

		#print results
		for i in range(len(compare_parameters)):
			print(f'{compare_parameters[i]}: rho = {corr_matrix[0][i]:0.03f}')

		### plot the correlation matrix
		fig, ax = plt.subplots(figsize = (5, 5))

		im = ax.imshow(corr_matrix, cmap = 'RdBu', origin = 'lower', vmin = -1, vmax = 1)

		#make colorbar
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size = '5%', pad = 0.05)

		cbar = plt.colorbar(im, cax = cax)
		cbar.ax.set_ylabel('Pearson correlation coefficient')

		tickpos = np.arange(corr_matrix.shape[0])

		ax.set_xticks(tickpos)
		ax.set_xticklabels(compare_parameters_names, rotation = 70, ha = 'right')
		ax.set_yticks(tickpos)
		ax.set_yticklabels(compare_parameters_names)

		ax.set_title('Correlation matrix for reproductive number R')

		plt.savefig(f'{R_plotloc}R_correlation_matrix.png', dpi = 200, bbox_inches = 'tight')
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
			'transit_smooth',
			'Rad',
			'TAvg',
			'HumAbsAvg'
		]

		#get the multiple parameters into a single array
		X = dataframes_to_NDarray(df_train, best_correlating_metrics)

		Y = np.array(df_train['Rt_avg'])
		#weights are important; can reduce training R^2 from 0.859 to 0.820
		weight = 1/np.array(df_train['Rt_abs_error'])

		###split into train and test set
		'''
		## perform simple elementwise random splitting
		rs = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 1923)

		#get splitting indices
		for train_index, test_index in rs.split(X, Y):
			pass

		#perform splitting
		X_train = X[train_index]
		Y_train = Y[train_index]
		weight_train = weight[train_index]

		X_test = X[test_index]
		Y_test = Y[test_index]
		weight_test = weight[test_index]
		'''

		## split off a test set being a continuous segment of data, which will give
		## a more robust evaluation
		#length of test segment in days
		test_set_days_length = 30
		test_set_days_length_timedelta = pd.Timedelta(f'{test_set_days_length} day')
		#get array of dates from which we will randomly pick a starting date
		startdate_array = df_train.loc[df_train.index < (df_train.index[-1] - test_set_days_length_timedelta)].index.values

		np.random.seed(1923)

		test_segment_startdate = np.random.choice(startdate_array, 1)[0]
		testmask = (df_train.index > test_segment_startdate) & (df_train.index <= (test_segment_startdate + test_set_days_length_timedelta))

		test_index = np.where(testmask)[0]

		X_train = dataframes_to_NDarray(df_train.loc[~testmask], best_correlating_metrics)
		Y_train = df_train.loc[~testmask]['Rt_avg'].values
		weight_train = 1/df_train.loc[~testmask]['Rt_abs_error'].values

		X_test = dataframes_to_NDarray(df_train.loc[testmask], best_correlating_metrics)
		Y_test = df_train.loc[testmask]['Rt_avg'].values
		weight_test = 1/df_train.loc[testmask]['Rt_abs_error'].values


		#apply ridge regression, see here for more info:
		#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
		if regression_method.lower() == 'ridge':
			clf = Ridge(alpha = 1)
		elif regression_method.lower() == 'linear':
			clf = LinearRegression()
		elif regression_method.lower() == 'adaboost':
			clf = AdaBoostRegressor()
		elif regression_method.lower() == 'randomforest':
			clf = RandomForestRegressor()
		else:
			raise ValueError('Incorrect regression method given')
		clf.fit(X_train, Y_train, sample_weight = weight_train)
		# clf.fit(X_train, Y_train)

		r_squared_train = clf.score(X_train, Y_train, sample_weight = weight_train)
		r_squared_test = clf.score(X_test, Y_test, sample_weight = weight_test)
		# r_squared = clf.score(X_train, Y_train)

		print(f'Train R^2: {r_squared_train:0.03f}')
		print(f'Test R^2: {r_squared_test:0.03f}')

		### plot the predictions versus ground truth
		fig, ax = plt.subplots()

		ax.scatter(Y, clf.predict(X), alpha = 0.4, color = 'navy', s = 8, label = f'Predictions (Test $R^2$ = {r_squared_test:0.03f})')
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

		plt.savefig(f'{R_plotloc}Mobility_R_prediction_accuracy.png', dpi = 200, bbox_inches = 'tight')
		plt.close()


		### now make and plot predictions
		X_pred = dataframes_to_NDarray(df_pred, best_correlating_metrics)

		df_pred['Rt_pred'] = clf.predict(X_pred)


		fig, ax1 = plt.subplots()

		ax2 = ax1.twinx()

		ln1 = ax1.plot(df_pred.index, df_pred.Rt_avg, label = 'Ground truth', color = betterblue)
		#indicate error margins on ground truth
		ax1.fill_between(df_pred.index, df_pred['Rt_low'], df_pred['Rt_up'], alpha = 0.4, color = betterblue)

		## plot prediction
		# ln2 = ax1.plot(df_pred.index, df_pred['Rt_pred'], label = f'Prediction (Test $R^2$: {r_squared_test:0.03f})', color = betterorange)
		#get test set mask for this new dataframe
		testmask_forplot = (df_pred.index > test_segment_startdate) & (df_pred.index <= (test_segment_startdate + test_set_days_length_timedelta))

		#train/pred
		#set values at test set to nan so that no line is plotted there
		df_pred_plot = df_pred.copy()
		df_pred_plot = df_pred_plot.set_value(df_pred_plot.loc[testmask_forplot].index, 'Rt_pred', np.nan)
		ln2 = ax1.plot(df_pred_plot.index, df_pred_plot['Rt_pred'], label = f'{regression_method} prediction without test set', color = betterorange)

		#test set
		ln3 = ax1.plot(df_pred.loc[testmask_forplot].index, df_pred.loc[testmask_forplot]['Rt_pred'], label = f'{regression_method} prediction test set ($R^2 = {r_squared_test:0.03f}$)', color = betterorange, linestyle = '--')


		#also plot government response
		#limit to days with prediction data
		df_response_plot = df_response_plot.loc[df_response_plot.index <= df_pred.index[-1]]
		ln4 = ax2.plot(df_response_plot.index, df_response_plot['StringencyIndex'], label = 'Stringency index', color = betterblack)

		ax1.grid(linestyle = ':')
		# ax.legend(loc = 'best')

		lns = ln1 + ln2 + ln3 + ln4
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc = 'lower left', prop = {'size': 8})

		ax1.set_ylabel('$R$')
		ax2.set_ylabel('Oxford Stringency Index')
		ax1.set_title(f'$R$ prediction up to {df_pred.index[-1].date()}')

		ax2.set_ylim(0)

		indicate_school_closed(ax1)

		#fix date ticks
		ax1.set_xticks(pd.date_range(np.min(df_pred.index), np.max(df_pred.index + pd.Timedelta(f'7 day')), freq = '2W'))
		fig.autofmt_xdate()
		myfmt = mdates.DateFormatter('%d-%m-%Y')
		ax1.xaxis.set_major_formatter(myfmt)

		plt.savefig(f'{R_plotloc}Mobility_R_prediction.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

def estimate_recent_prevalence(enddate_train = '2020-11-01', smoothsize = 5, regression_method = 'Linear'):
	"""
	Estimate the recent prevalence based on the test positivity ratio
	"""
	print(f'WARNING: end date for prevalence training set is {enddate_train}')

	startdate_train = '2020-09-08'

	df_prevalence, df_Rt = load_prevalence_Rt_data()
	df_overall_positive_tests = load_daily_covid(correct_for_delay = False)
	df_individual = load_individual_positive_test_data(load_agegroups = True)
	df_n_tests = load_number_of_tests(enddate = np.datetime64('today'))
	df_sewage = load_sewage_data(smooth = True, shiftdates = False)
	#for the plot as a reference
	df_response = load_government_response_data()

	print('------------')
	print('Prevalence prediction input data latest entries:')
	print('Overall positive tests data: ' + str(df_overall_positive_tests.index[-1].date()))
	print('Individual positive test data: ' + str(df_individual.index[-1].date()))
	print('N tests data: ' + str(df_n_tests.index[-1].date()))
	print('Sewage data: ' + str(df_sewage.index[-1].date()))
	print('------------')

	df_response = df_response.loc[df_response.index > startdate_train]

	agegroup_cols = list(df_individual.columns)
	#convert to percentages to make it a bit easier for the ML algorithms to learn
	df_individual[agegroup_cols] *= 100

	#correct the results to the day of the test
	df_overall_positive_tests.index = df_overall_positive_tests.index - pd.Timedelta(f'{result_delay} day')

	#merge datasets for test positivity calculations
	df_daily_covid = df_n_tests[['Number_of_tests']].join(df_overall_positive_tests[['Total_reported']], how = 'inner')

	#determine test positivity rate for the overall sample
	df_daily_covid['Positivity_ratio'] = df_daily_covid['Total_reported']/df_daily_covid['Number_of_tests']

	del df_daily_covid['Number_of_tests'], df_daily_covid['Total_reported']

	if smoothsize != None:
		#smooth the testing data, as there are some days with IT problems and underreported
		#tests, while having overreports the next few days.
		df_daily_covid['Positivity_ratio_smooth'] = df_daily_covid['Positivity_ratio'].rolling(smoothsize).mean()

		del df_daily_covid['Positivity_ratio']

		print('Using smoothed positivity data')

		#also smooth the day of disease onset data
		df_individual[agegroup_cols] = df_individual[agegroup_cols].rolling(smoothsize).mean()

	#correct the overall results to the day of infection
	df_daily_covid.index = df_daily_covid.index - pd.Timedelta(f'{int(incubation_period + time_to_test_delay)} day')

	### final dataset merge
	#merge overall positivity ratio with age group data
	df_daily_covid = df_individual.join(df_daily_covid, how = 'inner')
	#merge with sewage data
	df_predictors = df_daily_covid.join(df_sewage[['RNA_flow_smooth']], how = 'inner')

	print(df_predictors.columns)
	print('Combined data: ' + str(df_predictors.index[-1].date()))

	print('------------')

	#determine error on prevalence
	df_prevalence['prev_abs_error'] = ((df_prevalence['prev_low'] - df_prevalence['prev_avg']).abs() + (df_prevalence['prev_up'] - df_prevalence['prev_avg']).abs())/2

	# df_daily_covid['Total_per_million'] = df_daily_covid['Total_reported'] * per_million_factor


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

	### Choose parameters to use

	parameters_used = [
	'RNA_flow_smooth'
	] + agegroup_cols

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

	######## Finally finished with data merging

	### get data into the shape required for sklearn functions
	X = dataframes_to_NDarray(df_predictors_cor, parameters_used)
	Y = np.array(df_prevalence_cor['prev_avg'])
	weight = 1/np.array(df_prevalence_cor['prev_abs_error'])

	### plot correlation matrix
	if True:
		corr_matrix = np.corrcoef(np.concatenate((Y[:,None], X), axis = 1).T)

		compare_parameters_names = ['Prevalence'] + [param.replace('_smooth', '') for param in parameters_used]

		#print results
		print('Correlation of predictor parameters with prevalence:')
		for i in range(len(compare_parameters_names)):
			print(f'{compare_parameters_names[i]}: rho = {corr_matrix[0][i]:0.03f}')
		print('-------------')

		### plot the correlation matrix
		fig, ax = plt.subplots(figsize = (5, 5))

		im = ax.imshow(corr_matrix, cmap = 'RdBu', origin = 'lower', vmin = -1, vmax = 1)

		#make colorbar
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size = '5%', pad = 0.05)

		cbar = plt.colorbar(im, cax = cax)
		cbar.ax.set_ylabel('Pearson correlation coefficient')

		tickpos = np.arange(corr_matrix.shape[0])

		ax.set_xticks(tickpos)
		ax.set_xticklabels(compare_parameters_names, rotation = 70, ha = 'right')
		ax.set_yticks(tickpos)
		ax.set_yticklabels(compare_parameters_names)

		ax.set_title('Correlation matrix for prevalence and predictor variables')

		plt.savefig(f'{Prev_plotloc}Prevalence_correlation_matrix.png', dpi = 200, bbox_inches = 'tight')
		plt.close()

	###split into train and test set
	'''
	## use a random split
	rs = ShuffleSplit(n_splits = 1, test_size = 0.4, random_state = 1923)

	#get splitting indices
	for train_index, test_index in rs.split(X, Y):
		pass

	#perform splitting
	X_train = X[train_index]
	Y_train = Y[train_index]
	weight_train = weight[train_index]

	X_test = X[test_index]
	Y_test = Y[test_index]
	weight_test = weight[test_index]
	'''

	## split off a test set being a continuous segment of data, which will give
	## a more robust evaluation
	#length of test segment in days
	test_set_days_length = 20
	test_set_days_length_timedelta = pd.Timedelta(f'{test_set_days_length} day')
	#get array of dates from which we will randomly pick a starting date
	startdate_array = df_prevalence_cor.loc[df_prevalence_cor.index < (df_prevalence_cor.index[-1] - test_set_days_length_timedelta)].index.values

	np.random.seed(2000)

	test_segment_startdate = np.random.choice(startdate_array, 1)[0]
	testmask = (df_predictors_cor.index > test_segment_startdate) & (df_predictors_cor.index <= (test_segment_startdate + test_set_days_length_timedelta))

	test_index = np.where(testmask)[0]

	X_train = dataframes_to_NDarray(df_predictors_cor.loc[~testmask], parameters_used)
	Y_train = df_prevalence_cor.loc[~testmask]['prev_avg'].values
	weight_train = 1/df_prevalence_cor.loc[~testmask]['prev_abs_error'].values

	X_test = dataframes_to_NDarray(df_predictors_cor.loc[testmask], parameters_used)
	Y_test = df_prevalence_cor.loc[testmask]['prev_avg'].values
	weight_test = 1/df_prevalence_cor.loc[testmask]['prev_abs_error'].values


	### apply regression
	if regression_method.lower() == 'ridge':
		clf = Ridge(alpha = 1)
	elif regression_method.lower() == 'linear':
		clf = LinearRegression()
	elif regression_method.lower() == 'adaboost':
		clf = AdaBoostRegressor()
	elif regression_method.lower() == 'randomforest':
		clf = RandomForestRegressor()
	else:
		raise ValueError('Incorrect regression method given')
	clf.fit(X_train, Y_train, sample_weight = weight_train)
	# clf.fit(X_train, Y_train)

	r_squared_train = clf.score(X_train, Y_train, sample_weight = weight_train)
	r_squared_test = clf.score(X_test, Y_test, sample_weight = weight_test)

	print(f'Train R^2: {r_squared_train:0.03f}')
	print(f'Test R^2: {r_squared_test:0.03f}')
	print(f'Test set segment from {test_segment_startdate.astype("datetime64[D]")} to {np.datetime64(test_segment_startdate + test_set_days_length_timedelta).astype("datetime64[D]")}')


	### plot accuracy of predictions using the predictions versus ground truth
	fig, ax = plt.subplots()

	test_pred = clf.predict(X_test)
	ax.scatter(Y_test, test_pred, alpha = 0.4, color = 'navy', s = 8, label = f'Predictions (Test $R^2$ = {r_squared_test:0.03f})')

	#plot error bars
	errorbar_data = np.stack((df_prevalence_cor['prev_low'].values[test_index], df_prevalence_cor['prev_up'].values[test_index]))
	ax.errorbar(Y_test, test_pred, xerr = errorbar_data, ecolor = 'navy', elinewidth = 0.5, capthick = 0.5, capsize = 2, errorevery = 2, ls = 'none')


	#indicate one to one relationship
	xlims = ax.get_xlim()
	ylims = ax.get_ylim()

	startpoint = min((min(xlims), min(ylims)))
	endpoint = max((max(xlims), max(ylims)))

	ax.plot([startpoint, endpoint], [startpoint, endpoint], color = 'black', label = f'Ideal predictions ({regression_method.lower()} model)')
	ax.set_xlim(xlims)
	ax.set_ylim(ylims)

	ax.grid(linestyle = ':')
	ax.legend(loc = 'best')

	ax.set_xlabel('Measured prevalence [per million]')
	ax.set_ylabel('Predicted prevalence [per million]')

	ax.set_title('Prevalence prediction accuracy using mobility data')

	plt.savefig(f'{Prev_plotloc}Prevalence_prediction_accuracy.png', dpi = 200, bbox_inches = 'tight')
	plt.close()


	### Now make predictions for the most recent data
	#select the positive test data
	df_predictors_pred = df_predictors.loc[df_predictors.index > startdate_for_cor]
	# print(df_predictors_pred.tail(10))
	# print('^ data used for predictions ^')
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

	### predictions - plot test set separately
	#get test set mask for this new dataframe
	testmask_forplot = (df_prevalence_pred.index > test_segment_startdate) & (df_prevalence_pred.index <= (test_segment_startdate + test_set_days_length_timedelta))

	#train/pred
	#set values at test set to nan so that no line is plotted there
	df_prevalence_pred_plot = df_prevalence_pred.copy()
	df_prevalence_pred_plot = df_prevalence_pred_plot.set_value(df_prevalence_pred_plot.loc[testmask_forplot].index, 'Prev_pred', np.nan)
	ln2 = ax1.plot(df_prevalence_pred_plot.index, df_prevalence_pred_plot['Prev_pred'], label = f'{regression_method} prediction without test set', color = betterorange)

	#test set
	ln3 = ax1.plot(df_prevalence_pred.loc[testmask_forplot].index, df_prevalence_pred.loc[testmask_forplot]['Prev_pred'], label = f'{regression_method} prediction test set ($R^2 = {r_squared_test:0.03f}$)', color = betterorange, linestyle = '--')


	#also plot government response
	df_response_plot = df_response.loc[df_response.index <= df_prevalence_pred.index[-1]]
	ln4 = ax2.plot(df_response_plot.index, df_response_plot['StringencyIndex'], label = 'Stringency index', color = betterblack)

	ax1.grid(linestyle = ':')

	ax1.set_ylabel('Prevalence per million')
	ax2.set_ylabel('Oxford Stringency Index')

	ax1.set_title('COVID-19 estimated active cases in the Netherlands\nwith prediction of recent days')

	lns = ln1 + ln2 + ln3 + ln4
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc = 'lower right', prop = {'size': 8})

	indicate_school_closed(ax1)

	# ax.xaxis.set_tick_params(rotation = 45)
	# ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 3, maxticks = 6))
	#set location of date ticks, as the automatic technique does not work
	ax1.set_xticks(pd.date_range(np.min(df_prevalence_sel.index), np.max(df_prevalence_pred.index), freq = 'W'))

	fig.autofmt_xdate()
	myfmt = mdates.DateFormatter('%d-%m-%Y')
	ax1.xaxis.set_major_formatter(myfmt)

	ax1.set_ylim(0)
	ax2.set_ylim(0)

	plt.savefig(f'{Prev_plotloc}Prevalence_second_wave_with_predictions.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

def main():
	# government_response_results_simple()
	# plot_hospitalization()
	# stringency_R_correlation(enddate = '2021-01-07')
	# plot_superspreader_events()
	# plot_R_versus_weather()
	# plot_longterm_prevalence_decay(given_R = 0.9)

	epidemiological_modelling()

	# plot_daily_results(use_individual_data = True, startdate = '2020-09-01')
	# plot_prevalence_R()
	# plot_mobility()
	# plot_sewage()
	# plot_individual_data()
	# plot_cluster_change()

	# df = load_government_response_data(country = 'BEL')
	# pd.set_option('display.max_rows', None)
	# print(df['C6'])

	# estimate_recent_R(enddate_train = '2021-01-08', regression_method = 'Ridge')
	# estimate_recent_prevalence(enddate_train = '2021-01-13', regression_method = 'AdaBoost')

if __name__ == '__main__':
	main()
#
