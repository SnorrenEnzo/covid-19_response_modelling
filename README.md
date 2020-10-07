## Modelling the outcome of Dutch government coronavirus response


### Data sources

Number of contagious people (prevalence): [link](https://data.rivm.nl/covid-19/COVID-19_prevalentie.json)
Reproductive number: [link](https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json)

More raw data: [link](https://coronadashboard.rijksoverheid.nl/verantwoording)


#### Detailed statistics

Case mortality rate: [link](https://en.wikipedia.org/wiki/COVID-19_pandemic_death_rates_by_country)
- 4.6% for the Netherlands

### Modelling

Disease model: [SIRD](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model)
Integrator: [RK4](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
Basic reproduction number: [link](https://en.wikipedia.org/wiki/Basic_reproduction_number)

Other ways of modelling:
- [link](https://www.nas.ewi.tudelft.nl/index.php/coronavirus)
