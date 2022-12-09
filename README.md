<h1><center>Titanic in Production!</center></h1>

![titanic image](/assets/titanic.jpg "a big boat")

# Project Background
Suppose the year is 1912 and the Titanic disaster just occured. In a stroke of genius, the White Star Line company that operated the Titanic decided to launch
another Titanic the following year, but this time they hired a data scientist to try and predict ahead of time who would survive if there were another iceberg
catastrophe. Moreover, the company assured everyone that the new Titanic would be fitted with more life boats so that **more passengers could survive** if 
there were another catastrophe.

Unfortunately, it turns out that the next Titanic suffers the same fate as the original. But the White Star Line company is undeterred; they continue to manufacture, 
improve, and launch new Titanics every year, for 10 years total. How might we, as data scientists, predict who is likely to survive on each of these doomed voyages?

# Data Drift
The typical data science project in its simplest form involves a dataset and a metric to maximize/minimize. But when it comes to machine learning in production, it is 
often the case that the data used in training no longer reflects the data that is being seen in real life. In the example above, the fact that the White Star Line 
company is actively trying to make changes that lead to more people surviving each new Titanic is a form of **data drift**. 

For this project, the graph in ***Figure 1*** summarizes the drift that is occuring due to the modifications the White Star Line company is making:

| ![titanic image](/assets/titanic_survival_rates.png "increasing survival rates over time") | 
|:--:| 
| ***Figure 1*** |

To bring this back to our fictional scenario, this means that in 1913, around 50% of passengers survive; meanwhile, by the year 1922, around 85% of passengers survive.

How are we going to ensure good model performance over all of the years, from 1913 to 1922?

## Naive approach
We might be tempted to brag about our model that we trained on the original dataset and let it handle predictions over all the years. Maybe it even got a gold medal on 
Kaggle!

The problem is that in the original titanic dataset, only about 38% of passengers survived. Thus, any model trained on this data will learn that certain features lead 
to someone not surviving. These relationships just **won't be true anymore** since more people are being saved. 

However, the curious reader will observe that how we assign "who survives" in the new Titanics matters; after all, if I say that in the year 1923, all passengers age 
30+ get a life boat, age would become a very important feature in predicting survival! 

# The data generating process
 
The datasets for each of the new datasets were generated with CTGAN model presented in the paper *Modeling Tabular data using Conditional GAN* by Xu et al.[1]. The 
model was provided preprocessed data on the original Titanic and was asked to generate 10 similar datasets. Then, the increase in survival rates was added by sampling
from each generated dataset some fraction and assigning a label of "survived" to each observation in this sample. E.g. in the first dataset, the sample size was 10% of 
the dataset. In the final dataset, the sample size was 80%.

The preprocessing proccess went as follows:
- drop columns "PassengerId", "Name", "Ticket", and "Cabin"
- Impute the median (and in the case of categorical columns, the most frequent category) for any missing values (these columns were "Age", "Fare", and "Embarked")
- One-hot encode categorical columns, ie: Sex, Pclass, and Embarked
- Finally, standardize Age and Fare columns.

*Note that the code used to generate the datasets can be found in generate_datasets.py*

[1]: https://arxiv.org/abs/1907.00503

# A Better Approach to Handling Drift
Instead of following the naive approach, we might consider adding a trend feature to our datasets and re-training the model after each titanic goes out. This works 
especially because we know there's a linear trend in the numbers of passengers surviving (as described in the "data generating process" section -- the fraction of 
passengers being assigned "survived" is linearly increasing from 10% to 80% in year 7 onwards).

This simple idea of adding a trend and re-training the model pays massive dividends; consider the following graphs:

| ![titanic image](/assets/titanic_performance_accuracy.png "updating vs no updating") | 
|:--:| 
| ***Figure 2*** |

| ![titanic image](/assets/titanic_performance_f1.png "updating vs no updating") | 
|:--:| 
| ***Figure 3*** |

| ![titanic image](/assets/titanic_performance_logloss.png "updating vs no updating") | 
|:--:| 
| ***Figure 4*** |

In each of the graphs above, the model that was updated after each iteration with this Trend feature significantly outperforms the "no updating", aka Naive approach, 
model.

*The code that re-trains the model with the trend feature is found in app.py*

# Conclusion
The framework of the typical data science project does a slight disservice to beginners in that it doesn't emphasize the importance of considering **what might happen
to the model's performance in the real world with new data**. By taking the classic beginner Titanic project and extending it to a simulated "real world" environment, 
I hope that I've helped shift some mindsets to a more holistic view of data science projects and the types of interesting questions we can ask about "completed" data 
science projects.
