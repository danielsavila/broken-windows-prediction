<!-- ABOUT THIS PROJECT -->
# About The Broken Windows Project

The Broken Windows theory is a criminological theory popularized in the 1990's that says that visible signs of minor societal disorder, minor crime, and unkempt neighborhoods encouraged the future proliferation of more serious offenses. While the theory has largely been disproven by Steven Levitt and Jens Ludwig at the University of Chicago on a causal basis, minor crime, societal disorder, and unkempt neighborhoods could still be used as valuable correlates of crime in a neighborhood. This project attempts to determine whether graffiti cleanup and pothole filling in the City of Chicago are good indicators of crime prediction in a given neighborhood for a given month. 

This project takes data from the Chicago City Data Portal (https://data.cityofchicago.org/) and attempts to make predictions on next months crime rate using the previous months crime rate, the number of potholes filled, and the number of graffiti cleanup incidents within a given neighborhood. 

**Quick ethical disclosure** - While the intent of this project is good (attempting to build a data tool for effective government resource allocation), it is important to note that using the predictions of this model for police, school, violence intervention, or other resource allocation strategies has the potential for powerful positive and negative impacts on potentially vulnerable socioeconomic subpopulations within the City of Chicago. I have created this repo to demonstrate an analytical skillset, and it is the user's responsbility to make ethical choices regarding the use and interpretation of this analysis for public benefit.

<!-- OUTCOMES -->
## Best Model
RNN proved to be the best model, with a RMSE of ~2 on the test set. 

LSTM and GAN models were also tested (since their implementation is quite straight forward in Pytorch) but they were not as successful as RNN. In a way this makes sense - the connection between crimes/potholes/graffiti that occurred in February probably does not influence crime rates in, say, June. But crime/potholes/graffiti that occurs in February might have an impact in March or April, which means that the vanishing gradient problem does not seem to apply in this context in the way that it might for NLP problems, which LSTM and GAN are better for. 

XGBoost provided a strong initial starting point, with a RMSE of approximately ~30 crimes per month.

<!-- DATA-->
## About the Data

**Pothole Data:**  (https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Pot-Holes-Reported-Historical/7as2-ds3y/about_data)

The Chicago Department of Transportation is responsible for patching potholes in Chicago, and receives reports of potholes through the 311 call center from 2014 - 2019.

**Graffiti Data:** (https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Graffiti-Removal-Historical/hec5-y4x5/about_data)

Similar to the pothole data above, the Department of Streets and Sanitation are responsible for graffiti cleanup where they receive reports of graffiti through the 311 call center. Data from 2011 - 2025.

**Crime Data:** (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

Reported incidents of crime come from the Chicago Police Department's CLEAR system. This dataset contains data from 2001 to present, minus the most recent seven days, with the exception of murders.


<!-- ABOUT -->
# Why crime prediction is important:
The obvious benefits of crime prediction include the improved allocation of police resources to either prevent crime or improve potential repsonse times. However, other city departments would benefit from a crime prediction tool, for example the Chicago Department of Family and Support Services, Chicago Department of Public Health, Chicago Fire Department, and the Chicago Public School District. Crime often intersects with numerous socioeconomic factors, therefore introducing a targeted allocation of education opportunities, workforce development programs, overdose prevention services, and/or homelessness services to neighborhoods with need would be a benefit to all citizens of Chicago. 

## About this repo:
This repo evaluates 3 models. The models are the following...
* XGBoostRegressor
* Multilayer Perceptron
* Recurrent Neural Network

### Built With:

This project was built using the following libraries...
* Pandas
* Numpy
* Scikit-Learn
* DataBricks
* MatPlotLib
* Seaborn
* XGBoost
* Pytorch
* FastAPI
* Uvicorn

and packaged with Docker and FastAPI.

<!-- CONTACT -->
## Contact

Daniel Avila - (https://www.linkedin.com/in/daniel-avila-123392149/) - danielsavila2020@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
