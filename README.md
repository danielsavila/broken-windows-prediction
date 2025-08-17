<!-- ABOUT THIS PROJECT -->
# About The Broken Windows Project

The Broken Windows theory is a criminological theory popularized in the 1990's that says that visible signs of minor societal disorder, minor crime, and unkempt neighborhoods encouraged the future proliferation of more serious offenses. While the theory has largely been disproven by Steven Levitt at the University of Chicago on a causal basis, minor crime, societal disorder, and unkempt neighborhoods could still be used as valuable correlates of crime in a neighborhood. This project attempts to determine whether graffiti cleanup and pothole filling in the City of Chicago are good indicators of crime prediction in a given neighborhood for a given month. 

This project takes data from the Chicago City Data Portal (https://data.cityofchicago.org/) and attempts to make predictions on next months crime rate using the previous months crime rate, the number of potholes filled, and the number of graffiti cleanup incidents within a given neighborhood. 

**Quick ethical disclosure** - While the intent of this project is good (attempting to build a data tool for effective government resource allocation), it is important to note that using the predictions of this model for police, school, violence intervention, or other resource allocation strategies has the potential for powerful positive and negative impacts on potentially vulnerable socioeconomic subpopulations within the City of Chicago. I have created this repo to demonstrate an analytical skillset, and it is the user's responsbility to make ethical choices regarding the use and interpretation of this analysis for public benefit.


<!-- DATA-->

## About the Data

**Pothole Data**

**Graffiti Data**

**Crime Data**

## Best Model
The *XGBoost Classifier* performed the best of the 5 models evaluated, generating a 80% recall rate in the test set for positive attrition cases. In otherwords, of the people who actually left the firm, we captured 80% of them with this model. 

## Business Case, Using XGBoost Outcomes
As of August 2025, IBM employed 293,400 employees. For simplicity, we assume a company of employee size 300,000, an average annual attrition rate of 5%, the cost of replacing a full-time employee at $52,000, and the success rate of our retention program at 100%.

Precision for attrition cases in the test set was 24%, which means that of the people we classified as leaving, 24% of them actually left. Since 15,000 people leave every year, this implies that our model would predict that 62,500 people would be tagged as potentially leaving. 

As the cost of attrition is $780M (15,00 employees annually x $52,000 per employee), if the firm gave each of the 62,500 people who were tagged as potentially leaving no more than a **~$12,500 retention bonus, regardless of if the individual stayed,** the firm would break even on their recoverable* annual attrition costs by implementing this model.

**We say "recoverable" because recall is 80%, which means 20% of those that did leave were not captured by our model in the first place.


# Why is predicting attrition important:
* McKinsey estimates that the average cost of replacing a full time employee is $52,000, which incentivizes firms to get the hiring process right the first time. (https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/increasing-your-return-on-talent-the-moves-and-metrics-that-matter)
* While AI has eliminated numerous entry-level job openings, the opportunity cost of hiring the WRONG junior talent has increased as recruiters now need to filter through an exponentially increasing number of applicants.
* While the supply of inexperineced junior talent baloons, competition for experienced senior talent has increased as well, leading to higher pay packages, increased hiring timelines, and the potential for poaching from other firms.

If there was...
1) a way to identify individuals who were at high risk of leaving the firm, and 
2) a low cost intervention that increased the probability of that employee staying, then

there exists a break even point where the total cost of implementation and the total net savings are equal. Therefore, any implementation with total cost below that threshold would contribute net savings for the company. 

### i.e. if we can create a targeted retention program that, on average, costs less than losing an employee, we can save the firm some money, save years of intangible experience, and potentially maintain/improve company morale.

## About this repo:
This repo contains 5 .py files that contain different models fit to the data. The models are the following...
* K-Nearest Neighbors
* Linear/Quadratic Discriminant Analysis
* Logistic Regression
* Support Vector Machines (linear, polynomial, radial kernels)
* XGBoostClassifier


### Built With

This project was built using the following libraries...
* Pandas
* Numpy
* Scikit-Learn
* DataBricks
* Imbalanced-Learn
* MatPlotLib
* Seaborn
* XGBoost

and the following data science techniques...

--  Preprocessing
* Feature Scaling
* Random Oversampling
* Synthetic Data Generation (SMOTE)

--  Model Validation/Evaluation
* Cross Validation
* Confusion Matrices
* ROC/AUC Curves
* Hyperparameter Tuning (GridSearch)
* Evaluation Metrics (recall, f-1)
* Feature Importance

<!-- CONTACT -->
## Contact

Daniel Avila - (https://www.linkedin.com/in/daniel-avila-123392149/) - danielsavila2020@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
