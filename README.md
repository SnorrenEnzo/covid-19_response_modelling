## Modelling the outcome of Dutch government coronavirus response

### TODO:
- Auto download Apple/Google data

### Data sources

Number of contagious people (prevalence): [link](https://data.rivm.nl/covid-19/COVID-19_prevalentie.json)

Number of positively tested people in the Netherlands, updated each day at 10:00 CEST: [link](https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv)

Reproductive number: [link](https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json)

More raw data: [link](https://coronadashboard.rijksoverheid.nl/verantwoording)

A huge list of data sources: [link](https://www.databronnencovid19.nl/)

Weekly update [link](https://www.rivm.nl/documenten/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland) with:
- Number of tests performed per week
- Infection context, like at home, at work, pub etc.

#### Possible early predictors (for prevalence and R)

Mobility data:
- [Apple](https://covid19.apple.com/mobility)
- [Google](https://www.google.com/covid19/mobility/)

Sewage measurements:
- [data](https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv)
- [info](https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/a2960b68-9d3f-4dc3-9485-600570cd52b9?tab=relations)

#### Government response

[Overview](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker)

[Raw data](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv)

[Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) that indicates the meaning for various flags etc.

[Single country single day API](https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/actions/NLD/2020-09-20)

#### Detailed statistics
Incubation period
- 8.3 days, looks kind of exponential (or Poisson) [link](https://advances.sciencemag.org/content/6/33/eabc1202.full)

Case mortality rate: [link](https://en.wikipedia.org/wiki/COVID-19_pandemic_death_rates_by_country)
- 4.6% for the Netherlands

Days contagious after infection [link](https://www.theladders.com/career-advice/new-study-finds-covid-19-patients-remain-infectious-for-only-this-number-of-days)
- Up to the 11 day

##### Parameters of the SEIRD model:
Exposure period 1/a (NOT YET INFECTIOUS, average, exponential distribution with parameter a):
- Can't be really determined, let's set it to 2 days for now

Average time of infection 1/beta (serial interval)
- 4.7 days, based on 28 cases, looks like gamma distribution 4-2020 [link](https://www.sciencedirect.com/science/article/pii/S1201971220301193)
- 3.96 days, based on 468 cases 26-6-2020 [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7258488/)
- ~4.1 days, based on 1407 cases 18-6-2020 [link](https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciaa790/5859582)

Recovery time 1/gamma
- Time between infection and recovery: 14 days [link](https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf#:~:text=Using%20available%20preliminary%20data%2C,severe%20or%20critical%20disease.)

Mortality rate 1/mu
- Not mentioned a lot. Can however be calculated using 1/I dD/dt
	- dD/dt source: [link](https://www.rivm.nl/coronavirus-covid-19/grafieken)

### Data visualization
[covid-analytics.nl](https://covid-analytics.nl/population.html)

[CoronaWatchNL](https://github.com/Sikerdebaard/CoronaWatchNL)

### Modelling

Disease model, combination between these two:
- [SIRD](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model)
- [SEIR](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

Integrator: [RK4](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
Basic reproduction number: [link](https://en.wikipedia.org/wiki/Basic_reproduction_number)

Other ways of modelling:
- [link](https://www.nas.ewi.tudelft.nl/index.php/coronavirus)
