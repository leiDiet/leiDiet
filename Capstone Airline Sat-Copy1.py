#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Problem Statement
# 
# ## To prevent repeat customers from booking flights with their competition, Falcon airlines must maintain a high level of customer satisfaction.  Especially for those that travel frequently for business and pleasure. The main focus of this study is to find how Falcon passengers value each parameter from all listed in the satisfaction survey and try to quicky improve those offerings that are most important.
# 
# * Two separate data sets were collected from a random sample of 90,917 Falcon airline passengers.
# * The first data set provided the customers basic information along with their flights on-time performance.
# * Customer data included: Gender, Customer loyalty, Business or Personal travel, Cabin class, Flight distance, Departure and Arrival delays.
# * The second data set was a survey that was conducted post-flight asking customers to rate their overall flight experience. There were two rating options; satisfied and Neutral or dissatisfied. 
# * The second data set survey also asked them to rate their satisfaction for each parameter offered by the airline.  * There were six options available ranging  from extremely poor to excellent.
# * Rated parameters:  Seat comfort, Departure/Arrival time convenience, Food/drink, Gate Location, Inflight Wifi, Inflight Entertainment, Online support, Ease of Online Booking,  Onboard Service, Legroom, Baggage Handling,  Check-in service,  Cleanliness and Online boarding.
# 

# In[1]:


# Importing the necessary libraries
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries to tune model, get different metric scores and split data
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
)

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To build a logistic regression model
from sklearn.linear_model import LogisticRegression

# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To use statistical functions
import scipy.stats as stats

# To oversample and undersample data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# to perform PCA
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist


# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to perform PCA
from sklearn.decomposition import PCA

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To suppress the warning
import warnings

warnings.filterwarnings("ignore")

# This will help in making the Python code more structured automatically (good coding practice)
get_ipython().run_line_magic('load_ext', 'nb_black')


# In[2]:


# downloading first data set
data = pd.read_excel("Flight_data.xlsx")
print(f"There are {data.shape[0]} rows and {data.shape[1]} columns.")  # f-string

np.random.seed(1)  # To get the same random results every time


# In[3]:


# downloading second data set
data2 = pd.read_csv("Surveydata.csv")
print(f"There are {data2.shape[0]} rows and {data2.shape[1]} columns.")  # f-string

np.random.seed(1)  # To get the same random results every time


# In[4]:


# setting ID as index to combine with survey data
data.set_index("ID", inplace=True)
data.head()


# In[5]:


# getting a list of columns
cols = data.select_dtypes(["object"])
cols.columns


# In[6]:


# changing object dtypes to category
for i in cols.columns:
    data[i] = data[i].astype("category")


# In[7]:


cols_cat = data.select_dtypes(["category"])


# In[8]:


# looping category columns to find value counts for each variable
for i in cols_cat.columns:
    print("Unique values in", i, "are :")
    print(cols_cat[i].value_counts())
    print("-" * 50)


# In[9]:


# confirming datatypes changed to category
data.info()


# In[10]:


# matching column names to merge
data2.rename(columns={"Id": "ID"}, inplace=True)
data2.set_index("ID", inplace=True)
data2.head()


# In[11]:


# Checking for data types and missing values
data.info()


# In[12]:


# Checking for data types and missing values
data2.info()


# In[13]:


cols2 = data2.select_dtypes(["object"])
cols2.columns


# In[14]:


for i in cols2.columns:
    data2[i] = data2[i].astype("category")


# In[15]:


cols_cat2 = data2.select_dtypes(["category"])


# In[16]:


# looping category columns to find value counts for each variable
for i in cols_cat2.columns:
    print("Unique values in", i, "are :")
    print(cols_cat2[i].value_counts())
    print("-" * 50)


# In[17]:


# We need to merge the dataframe to one - inner merge using id to combine all records that match using customer
# id numbers
df_join = data.join(data2, how="inner")
np.random.seed(1)
df_join.sample(n=50)


# ## Summary of value counts
# * Most customers are loyal and travel on business
# * A slightly larger percentage of sample customers are satisfied with the airlines
# * Seat comfort and food and drink offerings are the two offereings that most customers are not satisfied with

# In[18]:


# checking for duplicates in the dataframe
df_join.duplicated().sum()


# In[19]:


# Renaming a few columns for shortening and clarity
df_join.rename(
    {
        "Departure.Arrival.time_convenient": "depart_arrivalTime.conv",
        "Inflightwifi_service": "Inflightwifi",
        "Leg_room_service": "Legroom",
        "Online_boarding": "Boarding",
    },
    axis=1,
    inplace=True,
)


# In[20]:


df_join["Gate_location"].replace(
    {
        "Convinient": "Convenient",
        "Inconvinient": "Inconvenient",
        "very convinient": "Very convenient",
        "very inconvinient": "Very inconvenient",
    },
    inplace=True,
)


# In[21]:


# double checking to make sure objects are changed to category
df_join.info()


# * There are 23 columns, all variables are categorical and numerical and memory usage reduced significantly.
# * It looks like there are 6 variables with missing data

# In[22]:


df_join.isnull().sum().sort_values(ascending=False)


# In[23]:


df_join.describe(include="all")


# There is a lot of missing data Females, Loyal Customers, Business Travels and Travelers flying business class respresent a larger portion of data.
# Most of the responses to each survey questions are acceptable to good.
# Mean age is 39

# In[24]:


# Making a copy of my dataframe
pass_data = df_join.copy


# # Univariate Analysis
# ## Numerical data

# In[25]:


# Univariate analysis of numerical variables to study central tendency and dispersion.
# Function to create boxplot and histogram for any input numerical variable.
# This function takes the numerical column as the input and returns the boxplots and histograms for the variable.


def histogram_boxplot(feature, figsize=(15, 10), bins=None):
    """Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        x=feature, kde=F, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        x=feature, kde=False, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        np.mean(feature), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        np.median(feature), color="black", linestyle="-"
    )  # Add median to the histogram


# In[26]:


histogram_boxplot(df_join["Age"])


# * Mean and median age is around 40 years old.
# * There are two peaks around 26 and 46.
# * Slightly right skewed

# In[27]:


histogram_boxplot(df_join["Flight_Distance"])


# * Flight distance is heavily right skewed
# * There appear to be a large number of outliers above 4000 miles
# * Mean and median both lie around 2000 miles

# In[28]:


histogram_boxplot(df_join["DepartureDelayin_Mins"])


# * Departure delays have a very large number of outliers
# * Mean and median are very low, meaning on average, the airlines mostly departs on time
# 

# In[29]:


histogram_boxplot(df_join["ArrivalDelayin_Mins"])


# * Arrival delays mirror departure delays

# ## Categorical Data

# In[30]:


def perc_on_bar(z):
    '''
    plot
    feature: categorical feature
    '''

    total = len(df_join[z]) # length of the column
    plt.figure(figsize=(15,5))
    ax = sns.countplot(df_join[z],palette='nipy_spectral')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot
        
        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage 
    plt.show() # show the plot


# In[31]:


perc_on_bar("Gender")


# * Both male and females are respresented equally in the dataset

# In[32]:


perc_on_bar("CustomerType")


# * Even with the 9K missing values, loyal customers represent a much larger sample in the dataset
# * What makes them loyal?

# In[33]:


perc_on_bar("Satisfaction")


# * A slightly higher percentage of passengers were satisified with their flights overall

# In[34]:


perc_on_bar("TypeTravel")


# * Similar to type of customer, business travelers represent a larger sample in this dataset.
# * There are also close to 9K missing values as well

# In[35]:


perc_on_bar("Class")


# * The dataset represents business class and economy class fairly evenly
# * We need further investigation to see if class has any correlation with satisfaction

# In[36]:


perc_on_bar("Seat_comfort")


# * More than half of the passengers were satisfied with the comfort of their seat.
# * Its possible this is class dependent and will be looked at further in the bivariate/multivariate analysis

# In[37]:


perc_on_bar("depart_arrivalTime.conv")


# * More than half of the passengers were satisfied with the departure and arrival time convenience
# * About 35% were not happy 
# * There are around 8,000 missing values in this category

# In[38]:


perc_on_bar("Food_drink")


# * More than half of the passengers were satisfied with food and drink offerings
# * About 38 percent were not happy with the offerings
# * There are around 8,000 missing values in this category

# In[39]:


perc_on_bar("Gate_location")


# * About 50 percent of the passengers feel that gate location is inconvenient or needs improvement
# * The other 50 percent say its manageable or convenient

# In[40]:


perc_on_bar("Inflightwifi")


# * A larger percentage of customers are content or happy with inflight wifi
# * Only 30 percent are unhappy

# In[41]:


perc_on_bar("Inflight_entertainment")


# * Similar to Wifi, 73 percent of sample passengers are happy with inflight entertainment

# In[42]:


perc_on_bar("Onboard_service")


# * Most passengers are content or happy with onboard services
# * There are around 7,000 missing values in this column

# In[43]:


perc_on_bar("Online_support")


# In[44]:


perc_on_bar("Ease_of_Onlinebooking")


# * Most customers are happy with online support and ease of online booking

# In[45]:


perc_on_bar("Legroom")


# * Most passengers are satisfied with leg room.

# In[46]:


perc_on_bar("Baggage_handling")


# * Most customers are happy with baggage handling.
# 

# In[47]:


perc_on_bar("Checkin_service")


# * Most customers are happy with check-in service

# In[48]:


perc_on_bar("Cleanliness")


# * Most customers are happy with cleaniless of the aircraft
# * This appears to be the service that has the least poor ratings

# In[49]:


perc_on_bar("Boarding")


# * Most customers are happy with the boarding process

# ## Summary 
# * There are a large number of outliers in flight distance and departure and arrival delays
# * Cleanliness and baggage handling received the highest ratings and the least amount of poor ratings
# * Gate location had the most negative feedback
# * Seat comfort and food/drink offerings also had more negative ratings, which could be equated to class
# * A large percentage of passengers in this dataset are loyal customers traveling for business, however there is around 9000 entries in these two categories that are missing. 
# * About half of the passengers in this dataset are traveled in business class

# # Bivariate Analysis

# In[50]:


# Transforming categorical variable into numeric, 1=neutral/dissatisfied 0=satisfied

from sklearn.preprocessing import LabelEncoder


# In[51]:


df_join["Satisfaction"] = np.where(
    df_join["Satisfaction"].str.contains("neutral or dissatisfied"), 1, 0
)


# In[52]:


df_join.Satisfaction.dtype


# In[53]:


df_join.Satisfaction.value_counts()


# In[54]:


df_join.head()


# In[55]:


sns.pairplot(df_join, hue="Satisfaction")


# In[56]:


plt.figure(figsize=(15, 7))
sns.heatmap(df_join.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="BuPu")
plt.show()


# In[57]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = df_join[predictor].nunique()
    sorter = df_join[target].value_counts().index[-1]
    tab1 = pd.crosstab(df_join[predictor], df_join[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    sns.set(palette="nipy_spectral")
    tab = pd.crosstab(
        df_join[predictor], df_join[target], normalize="index"
    ).sort_values(by=sorter, ascending=False)
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[58]:


stacked_barplot(df_join, "CustomerType", "Satisfaction")


# * Its very clear from this graph that loyal customers are more satisfied than disloyal

# In[59]:


stacked_barplot(df_join, "Gender", "Satisfaction")


# * Women are more satisfied than men: 65% vs 44%

# In[60]:


stacked_barplot(df_join, "TypeTravel", "Satisfaction")


# * Business travelers were more satisfied with their flights than the personal travelers

# In[61]:


stacked_barplot(df_join, "Class", "Satisfaction")


# * Business class travelers are happier overall than economy class travelers
# * 70% satisified versus 40%

# In[62]:


stacked_barplot(df_join, "Seat_comfort", "Satisfaction")


# * Surprisingly, the passengers that rated seat comfort as extremely poor were mostly satisifed overall.
# * Travelers that rated the seats excellent were also mostly satisfied.
# * Passengers rating the seat comfort as acceptable and needs improvement were among the least satisfied in this group

# In[63]:


stacked_barplot(df_join, "depart_arrivalTime.conv", "Satisfaction")


# * There is nothing really noticeable among satisfaction ratings and convenience of departure and arrival time

# In[64]:


stacked_barplot(df_join, "Food_drink", "Satisfaction")


# * This variable is surprising in the same way seat comfort was.  80% of the passengers that gave an extremely poor rating were still overall satisfied on their flight.

# In[65]:


stacked_barplot(df_join, "Gate_location", "Satisfaction")


# * very inconvenient only had one response, so we can't count that as statistcally significant
# * Those that said the location was very convenient were most satisfied
# * Passengers that said the gate location was manageable and convenient were least satisfied overall.

# In[66]:


stacked_barplot(df_join, "Inflightwifi", "Satisfaction")


# * Passengers with good and excellent wifi experience were generally more satisified with their flight.
# * Passengers who rated the wifi as poor were least satisfied overall
# * The remaining passenger ratings were similar in scale, satisfaction around 50-55%

# In[67]:


stacked_barplot(df_join, "Inflight_entertainment", "Satisfaction")


# * Those who rated good and excellent were more satisfied overall on their flight
# *  Around 65% of the passengers entertainment with an extremely poor rating were still overall satisfied with their flight
# * Customers who gave acceptable, needs improvement and poor ratings overwhelmingly gave neutral/dissatisfied satisfaction ratings

# In[68]:


stacked_barplot(df_join, "Online_support", "Satisfaction")


# * Online support shows very polarizing data
# * Good and excellent ratings show 70-80% satisfaction
# * Acceptable and lesser ratings show very high amounts of neutral/dissatisfaction, around 75%.

# In[69]:


stacked_barplot(df_join, "Ease_of_Onlinebooking", "Satisfaction")


# * This variable shows a very similar output as online support.

# In[70]:


stacked_barplot(df_join, "Onboard_service", "Satisfaction")


# * This again is very similar to the two previous cell outputs

# In[71]:


stacked_barplot(df_join, "Legroom", "Satisfaction")

* Customers who had more legroom were generally more satisfied
* Passengers who rated extremely poor legroom were strangely more satisfied than other customers who gave poor to acceptable ratings
# In[72]:


stacked_barplot(df_join, "Baggage_handling", "Satisfaction")


# * Passengers that had a good/excellent experience with their baggage tend to be satisfied overall
# * Those who rated baggage handling as acceptable were least satisfied

# In[73]:


stacked_barplot(df_join, "Checkin_service", "Satisfaction")


# * Passengers that had a good/excellent experience with checkin tend to be satisfied overall
# * Those who rated checkin as needs improvement to extremely poor were least satisfied

# In[74]:


stacked_barplot(df_join, "Cleanliness", "Satisfaction")


# * Customers that gave a good/excellent rating for cleaniness were most satisfied overall
# * Acceptable rating customers were least satisfied

# In[75]:


stacked_barplot(df_join, "Boarding", "Satisfaction")


# * Passengers that rated the boarding process good/excellent were satifisied overall
# * Needs improvement to extremely poor ratings showed overall customer neutrality or dissatisfaction

# In[76]:


sns.lineplot(x="Age", y="Satisfaction", data=df_join)


# ## Summary
# * Loyal customers, which represent the larger class in the dataset are 20% more satisfied over disloyal
# * Females tend to be 20% more satisfied than men
# * Business travelers and business class passengers are more satisfied than personal travlers flying economy
# * As expected, most customers that gave a good/excellent rating for each of the survey parameters were satisfied overall
# * There were a few variables where an extremely poor rating was given but the overall satisfaction was positive: seat comfort, food and drink, inflight entertainment and legroom.
# * Most passengers that submitted needs improvement/poor ratings were more likely to be neutral or dissatisfied overall
# * Customers that submitted acceptable ratings varied between variables.  For example, online support and inflight entertainment = acceptable = neutral/dissatisfied overall.
# 
# * From only visual analysis, the attributes that seem to contribute most to overall satisfaction importance are seat comfort, inflight entertainment, online support, ease of online booking and boarding process.
# 

# * The customers most satisfied overall tend to be between 40 and 60 years old
# * Passengers neutral or least satisfied are in the 70+ age group

# # Data Pre-processing
# * This section will be about cleaning and processing data. This will include dropping data we don't need, adding new features, dealing with missing values and outliers.
# * We will start this with apparent drops and a little more eda.
# * We will then do a little more EDA with clean data to see if there are any new insights.

# In[77]:


# First we will drop the column arrival delay in minutes because it is almost perfectly correlated with departure
# delay and unlike departure delay, has 284 missing values
df_join.drop("ArrivalDelayin_Mins", axis=1, inplace=True)


# In[78]:


# We will also drop Customer Type as it doesn't seem to add much insight into the data and has a lot of missing data
df_join.drop("CustomerType", axis=1, inplace=True)


# In[79]:


df_join.sample(200)


# In[80]:


# Going to look at the rest of the missing values again
df_join.isnull().sum().sort_values(ascending=False)


# There is still quite a bit of missing information, so need to further investigate how to impute the missingness

# * Most of the data is from the 35 - 60 age group. 

# In[81]:


num_missing = df_join.isnull().sum(axis=1)
num_missing.value_counts()


# In[82]:


df_join[num_missing == 2].sample(n=5)


# In[83]:


for n in num_missing.value_counts().sort_index().index:
    if n > 0:
        print(f"For the rows with exactly {n} missing values, NAs are found in:")
        n_miss_per_col = df_join[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print("\n\n")


# There are no patterns of missing values per category that can help with imputation or deletion, we will impute values with most frequent, except for Type of Travel.  All children under 18 will be put under personal travel and not business.

# In[84]:


df_join.TypeTravel.nunique()


# In[85]:


df_join.Food_drink.nunique()


# In[86]:


df_join.Onboard_service.nunique()


# In[87]:


df_join.Onboard_service.nunique()


# In[88]:


"""Due to the nature of surveys and age, it is best to bin into age groups
 7 to 18 year olds are going to provide very different input than 35 -50 year olds and so on
There is also a difference in satisfaction among certain ages that we can group together as well
First we find the min and max"""

min_value = df_join['Age'].min()
max_value = df_join['Age'].max()
print(min_value)
print(max_value)


# In[89]:


# binning age groups to see if we can lose unwanted data
df_join["age_bin"] = pd.cut(
    x=df_join["Age"],
    bins=[6, 18, 26, 36, 66, 86],
    labels=["7 to 18", "19 to 25", "26 to 35", "36 to 60", "60 to 85"],
)
df_join.sample(50)


# In[90]:


df_join["age_bin"].value_counts()


# In[91]:


# It shows that there are 3345 7 to 18 year olds that are traveling on business
stacked_barplot(df_join, "age_bin", "TypeTravel")


# In[92]:


stacked_barplot(df_join, "age_bin", "Satisfaction")


# In[93]:


df_join.age_bin.dtype


# In[94]:


df_join["TypeTravel"] = (
    df_join["TypeTravel"]
    .astype(str)
    .replace("nan", "Business travel")
    .astype("category")
)


# In[95]:


df_join["TypeTravel"].isnull().sum()


# In[96]:


df_join.loc[df_join["Age"] < 19, "TypeTravel"] = "Personal Travel"


# In[97]:


# filling in the remaining missing values
df_join = df_join.fillna(df_join.mode().iloc[0])


# In[98]:


pd.crosstab(df_join["age_bin"], df_join["Satisfaction"])


# In[99]:


stacked_barplot(df_join, "age_bin", "Satisfaction")


# In[100]:


pd.crosstab(df_join["age_bin"], df_join["TypeTravel"])


# Binning and age/travel types are all taken care of.  We can now label encode travel type.

# In[101]:


df_join["TypeTravel"] = np.where(
    df_join["TypeTravel"].str.contains("Personal Travel"), 1, 0
)


# In[102]:


df_join.TypeTravel.value_counts()


# In[103]:


# checking to make sure missing values are taken care of
df_join.isnull().sum()


# 7 to 18 year olds traveling for business is highly unlikely so we will need to change that information

# * Again, here we can clearly see here that most of the unsatisfied/neutral customers are flying economy.
# * Though there are a lot of unsatisfied passengers at flight distances between 1200 and 3000 miles, there are still a number of satisfied customers, but as the flight distance increases, the less satisified customers appear

# In[104]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Seat_comfort",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[105]:


sns.catplot(
    x="Class",
    hue="Satisfaction",
    col="Legroom",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[106]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Legroom",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[107]:


sns.catplot(
    x="Seat_comfort",
    y="Flight_Distance",
    hue="Satisfaction",
    data=df_join,
    kind="bar",
)


# In[108]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Online_support",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[109]:


sns.catplot(x="age_bin", hue="Satisfaction", col="Class", data=df_join, kind="count")


# * Business class - business travelers are the most satisifed out of all the passengers
# * Economy class customer are most dissatisfied over all
# * Eco plus passengers are mostly split, with dissatisifaction slightly ahead

# In[110]:


sns.catplot(
    x="Class",
    hue="Satisfaction",
    col="Inflight_entertainment",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[111]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Inflightwifi",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[112]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Inflight_entertainment",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[113]:


sns.catplot(
    x="Onboard_service",
    hue="Satisfaction",
    col="age_bin",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[114]:


sns.catplot(
    x="age_bin",
    hue="Satisfaction",
    col="Food_drink",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# In[115]:


sns.catplot(
    x="Checkin_service",
    hue="Satisfaction",
    col="age_bin",
    col_wrap=3,
    data=df_join,
    kind="count",
)


# ## Summary
# * Seat comfort was an issue for the 65 to 80 year olds in particular
# * Inflight entertainment was important from ages 7 to 35
# * Onboard service and Food and drink were also important for under 35 and over 65
# * Boarding was important to 65 – 80 years
# * Check-in service, Wifi and Online support was important for all age groups (except 7-18)
# 

# In[116]:


df_join["Seat_comfort"].value_counts()


# In[117]:


le = LabelEncoder()


# In[118]:


order1 = {
    "extremely poor": 1,
    "poor": 2,
    "need improvement": 3,
    "acceptable": 4,
    "good": 5,
    "excellent": 6,
}


# In[119]:


df_join["Seat_comfort"].map(order1)


# In[120]:


df_join["Seat_enc"] = df_join["Seat_comfort"].map(order1)


# In[121]:


df_join.head()


# In[122]:


df_join["Food_drink"].map(order1)


# In[123]:


df_join["Food_drink_enc"] = df_join["Food_drink"].map(order1)


# In[124]:


df_join["Inflightwifi"].map(order1)


# In[125]:


df_join["wifi_enc"] = df_join["Inflightwifi"].map(order1)


# In[126]:


df_join["Inflight_entertainment"].map(order1)


# In[127]:


df_join["entertainment_enc"] = df_join["Inflight_entertainment"].map(order1)


# In[128]:


df_join["Onboard_service"].map(order1)


# In[129]:


df_join["Onboard_enc"] = df_join["Onboard_service"].map(order1)


# In[130]:


df_join["Legroom"].map(order1)


# In[131]:


df_join["Leg_enc"] = df_join["Legroom"].map(order1)


# In[132]:


df_join["Cleanliness"].map(order1)


# In[133]:


df_join["Clean_enc"] = df_join["Cleanliness"].map(order1)


# In[134]:


df_join["Online_support"].map(order1)


# In[135]:


df_join["Online_sup_enc"] = df_join["Online_support"].map(order1)


# In[136]:


df_join["Ease_of_Onlinebooking"].map(order1)


# In[137]:


df_join["Online_ease_enc"] = df_join["Ease_of_Onlinebooking"].map(order1)


# In[138]:


df_join["Baggage_handling"].map(order1)


# In[139]:


df_join["Baggage_enc"] = df_join["Baggage_handling"].map(order1)


# In[140]:


df_join["Checkin_service"].map(order1)


# In[141]:


df_join["Checkin_enc"] = df_join["Checkin_service"].map(order1)


# In[142]:


df_join["depart_arrivalTime.conv"].map(order1)


# In[143]:


df_join["dep_arr_conv_enc"] = df_join["depart_arrivalTime.conv"].map(order1)


# In[144]:


df_join.Gate_location.value_counts()


# In[145]:


order2 = {
    "Very inconvenient": 1,
    "Inconvenient": 2,
    "need improvement": 3,
    "manageable": 4,
    "Convenient": 5,
    "Very convenient": 6,
}


# In[146]:


df_join["Gate_location"].map(order2)


# In[147]:


df_join["Gate_loc_enc"] = df_join["Gate_location"].map(order2)


# In[148]:


df_join["Gate_loc_enc"].value_counts()


# In[149]:


df_join["Boarding"] = df_join["Boarding"].map(order1)


# In[150]:


Pre = df_join[
    [
        "Flight_Distance",
        "DepartureDelayin_Mins",
        "Online_ease_enc",
        "Online_sup_enc",
        "Gate_loc_enc",
        "Checkin_enc",
        "Boarding",
    ]
]
Pre


# In[151]:


df_join.Baggage_enc = df_join.Baggage_enc.astype("float")
df_join.Seat_enc = df_join.Seat_enc.astype("float")
df_join.Food_drink_enc = df_join.Food_drink_enc.astype("float")
df_join.entertainment_enc = df_join.entertainment_enc.astype("float")
df_join.Onboard_enc = df_join.Onboard_enc.astype("float")
df_join.Leg_enc = df_join.Leg_enc.astype("float")
df_join.Clean_enc = df_join.Clean_enc.astype("float")
df_join.Online_ease_enc = df_join.Online_ease_enc.astype("float")
df_join.Online_sup_enc = df_join.Online_sup_enc.astype("float")
df_join.Checkin_enc = df_join.Checkin_enc.astype("float")
df_join.dep_arr_conv_enc = df_join.dep_arr_conv_enc.astype("float")


# In[152]:


post = df_join["Baggage_enc"].mean()


# In[153]:


df_join.wifi_enc = df_join.wifi_enc.astype("float")


# In[154]:


Split_survey_to_cols = {'Preflight': ["Online_ease_enc","Online_sup_enc","Gate_loc_enc",
        "Checkin_enc","Boarding"],
        'Inflight': ["Seat_enc","Food_drink_enc","wifi_enc","entertainment_enc","Onboard_enc","Leg_enc","Clean_enc"]}

for Split_survey, colvec in Split_survey_to_cols.items():
    df_join[Split_survey] = round(df_join[colvec].mean(axis=1))


# In[155]:


plt.figure(figsize=(15, 7))
sns.heatmap(df_join.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="BuPu")
plt.show()


# ## Summary
# * The highest correlations among individual ranked variables are ease of online booking, online support and inflight wifi: baggage handling and Cleanliness: Seat comfort and food and drink: Onboard services and cleanliness.
# * Overall neutrality or dissatisfaction correlated negatively with most variables, inflight variables a little more.
# 
# 

# In[156]:


stacked_barplot(df_join, "Inflight", "Satisfaction")


# In[157]:


stacked_barplot(df_join, "Preflight", "Satisfaction")


# In[158]:


sns.catplot(
    x="Inflight",
    y="Flight_Distance",
    hue="Satisfaction",
    data=df_join,
    kind="box",
)


# In[159]:


sns.catplot(
    x="Preflight",
    y="Flight_Distance",
    hue="Satisfaction",
    data=df_join,
    kind="box",
)


# * Median about the same for all distances, but alot of outliers
# * A little more neutrality/dissatisfaction coming from the preflight columns 
# * Will treat outliers next

# In[160]:


sns.catplot(
    x="Inflight",
    y="DepartureDelayin_Mins",
    hue="Satisfaction",
    data=df_join,
    kind="box",
)


# * Flight distance may not be that relevant, at least it doesn't seem so much in the EDA, we will still treat it as important as departure delays.
# * Departure delays is heavily skewed and so many outliers, but seems to be an important variable, thus needs to be treated carefully.

# ## Outlier Treatment
# * Even though I will be performing Decision tree models, which are not sensitive to outliers, I may perform logistic regression and Clustering methods, therefore treating the outliers.

# In[161]:


## Departure delay is highly skewed and will use log transformation to assist with the skewness
plt.hist(df_join["DepartureDelayin_Mins"], bins=50)


# In[162]:


cols_to_log = ["DepartureDelayin_Mins", "Flight_Distance"]
for colname in cols_to_log:
    plt.hist(df_join[colname], bins=50)
    plt.title(colname)
    plt.show()
    print(np.sum(df_join[colname] <= 0))


# In[163]:


plt.hist(np.log(df_join['DepartureDelayin_Mins'] + 1), 50)
plt.title('log(Departure Delay + 1)')
plt.show()
plt.hist(np.arcsinh(df_join['DepartureDelayin_Mins']), 50)
plt.title('arcsinh(Departure Delay)')
plt.show()
plt.hist(np.sqrt(df_join['DepartureDelayin_Mins']), 50)
plt.title('sqrt(Departure Delay)')
plt.show()


# * The standard scaler assumes features are normally distributed and will scale them to have a mean 0 and standard deviation of 1.

# In[164]:


# Log Transformation has definitely helped in reducing the skew
# Creating a new column with the transformed variable.
df_join["DepartureDelayin_Mins"] = np.log(df_join["DepartureDelayin_Mins"])


# In[165]:


df_join[["Flight_Distance"]] = MinMaxScaler().fit_transform(
    df_join[["Flight_Distance"]]
)

df_join["Flight_Distance"].hist(bins=20)
plt.title("Flight Distance")
plt.show()


# # Defining dependent  and independent variables

# In[166]:


print(df_join.columns)


# In[167]:


X = df_join.drop(
    [
        "DepartureDelayin_Mins",
        "Satisfaction",
        "Seat_comfort",
        "depart_arrivalTime.conv",
        "Food_drink",
        "Gate_location",
        "Inflightwifi",
        "Inflight_entertainment",
        "Online_support",
        "Ease_of_Onlinebooking",
        "Onboard_service",
        "Legroom",
        "Baggage_handling",
        "Checkin_service",
        "Cleanliness",
        "Boarding",
    ],
    axis=1,
)

y = df_join[["Satisfaction"]]

print(X.head())
print(y.head())


# ## Endcoding the last categorical variables and splitting the train and test sets

# In[168]:


# encoding the categorical variables
X = pd.get_dummies(X, columns=["Gender", "Class", "age_bin"], drop_first=True)
X.head()


# In[169]:


print(X.shape)
print(y.shape)


# In[170]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[171]:


X_train.head()


# In[172]:


print(X_train.shape)


# * Because we have a target variable, 'Satisfaction', I will be primarily be building supervised learning models, decision trees and logistin regression.
# * I will experiment with K-means clustering as well because of the different segments in the customer data.

# # Milestone 3

# ## After more careful consideration and research, I have decided that Decision tree classification is the model more suited for this particular data set.  Here is why:
# 
# * Target is a binary categorical feature
# * Most features are categorical
# * non-linear relationships among features
# 
# Bagging and/or boosting using Decision tree criteria will help build an even more robust model
# 
# I will still first create and run a logistic regression model to compare.
# 

# ## Model evaluation criterion
# * I will be using Recall as a metric for our model performance, because here the company could face 2 issues: 
#     Predicting a person is satisfied when they are not.
#     Predicting a person is not satisfied, when in fact they are.
# * Which case is more important ?
#     Predicting the customer is dissatisfied as we need to find out who those people are and try to fix the issues that make them unhappy.
# * How to reduce this loss? 
#     Need to reduce False Negatives
#     
# Company wants Recall to be maximized, greater the recall lesser the chances of false negatives

# ## Logistic Regression
# 
# * Downloading necessary libraries to perform lg regressions.
# 
# * Defining confusion matrix to repeat metric performance on different models.
# 

# In[173]:


from sklearn.metrics import classification_report, confusion_matrix


def make_confusion_matrix(y_actual, y_predict, labels=[1, 0]):
    """
    y_predict: prediction of class
    y_actual : ground truth
    """
    cm = confusion_matrix(y_predict, y_actual, labels=[1, 0])
    df_cm = pd.DataFrame(
        cm, index=[i for i in ["1", "0"]], columns=[i for i in ["1", "0"]]
    )
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(7, 5))
    sns.heatmap(df_cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ## Multicollinearity
# * Variance Inflation factor: Variance inflation factors measure the inflation in the variances of the regression coefficients estimates due to collinearities that exist among the predictors. 
# * Measuring Satisfaction constant against variable coeffiecients
# * Dropping highest p values first

# In[174]:


import statsmodels.api as sm

# adding constant to training and test set
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

logit = sm.Logit(y_train, X_train)

lg = logit.fit()

print(lg.summary())

# Let's Look at Model Performance
y_pred = lg.predict(X_train)
pred_train = list(map(round, y_pred))

y_pred1 = lg.predict(X_test)
pred_test = list(map(round, y_pred1))

print("recall on train data:", recall_score(y_train, pred_train))
print("recall on test data:", recall_score(y_test, pred_test))


# * The outputs are not that bad but do need to be improved.  We will check for mulitcolinearity.

# In[175]:


# Downloading variance influation factor and creating a series to test

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_series1 = pd.Series(
    [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])],
    index=X_train.columns,
)
print("Series before feature selection: \n\n{}\n".format(vif_series1))


# There are some correlated variables here, which is to be understood. 
# Age, age bin 36 - 60 and type travel have high correlation and p values.
# We will remove variables slowly, starting with age and type travel.
# 

# In[176]:


# Age has high p-value and correlated with other variables. We start with there
X_train1 = X_train.drop("Age", axis=1)
X_test1 = X_test.drop("Age", axis=1)

logit1 = sm.Logit(y_train, X_train1)
lg1 = logit1.fit()
print(lg1.summary2())

# Let's Look at Model Performance
y_pred = lg1.predict(X_train1)
pred_train = list(map(round, y_pred))

y_pred1 = lg1.predict(X_test1)
pred_test = list(map(round, y_pred1))

print("Recall on train data:", recall_score(y_train, pred_train))
print("Recall on test data:", recall_score(y_test, pred_test))


# Not much change in recall.  Type Travel has the highest p-value.  We will drop that variable next to see if there is any change.

# In[177]:


# TypeTravel has highest p values among those with p-value greater than 0.05
X_train2 = X_train1.drop("TypeTravel", axis=1)
X_test2 = X_test1.drop("TypeTravel", axis=1)

logit2 = sm.Logit(y_train, X_train1)
lg2 = logit2.fit()
print(lg2.summary2())

# Let's Look at Model Performance
y_pred = lg2.predict(X_train1)
pred_train = list(map(round, y_pred))

y_pred1 = lg2.predict(X_test1)
pred_test = list(map(round, y_pred1))

print("recall on train data:", recall_score(y_train, pred_train))
print("recall on test data:", recall_score(y_test, pred_test))


# Still no change.  Will try to drop cleanliness next.

# In[178]:


# TypeTravel has highest p values among those with p-value greater than 0.05
X_train3 = X_train2.drop("Clean_enc", axis=1)
X_test3 = X_test2.drop("Clean_enc", axis=1)

logit3 = sm.Logit(y_train, X_train3)
lg3 = logit3.fit()
print(lg3.summary2())

# Let's Look at Model Performance
y_pred = lg3.predict(X_train3)
pred_train = list(map(round, y_pred))

y_pred1 = lg3.predict(X_test3)
pred_test = list(map(round, y_pred1))

print("recall on train data:", recall_score(y_train, pred_train))
print("recall on test data:", recall_score(y_test, pred_test))


# In[179]:


# TypeTravel has highest p values among those with p-value greater than 0.05
X_train4 = X_train3.drop("age_bin_19 to 25", axis=1)
X_test4 = X_test3.drop("age_bin_19 to 25", axis=1)

logit4 = sm.Logit(y_train, X_train4)
lg4 = logit4.fit()
print(lg4.summary2())

# Let's Look at Model Performance
y_pred = lg4.predict(X_train4)
pred_train = list(map(round, y_pred))

y_pred1 = lg4.predict(X_test4)
pred_test = list(map(round, y_pred1))

print("recall on train data:", recall_score(y_train, pred_train))
print("recall on test data:", recall_score(y_test, pred_test))


# We will try to drop one more with p-value at 3.077 to see if there is any changes

# * Not much has changed when dropping mulitple variables with mulitcollinearity.  
# * We can use any of the models as they have approximately the same recall values.
# * We will go with lg2, where we removed of age and Travel Type.  Removing certain categories from a variable will impact the interpretations from the model.  Age bin and Cleanliness are those such variables. 
# 

# In[180]:


# Let's Look at Model Performance
y_pred = lg2.predict(X_train1)
pred_train = list(map(round, y_pred))

y_pred1 = lg2.predict(X_test1)
pred_test = list(map(round, y_pred1))


# In[181]:


print("Accuracy on train data:", accuracy_score(y_train, pred_train))
print("Accuracy on test data:", accuracy_score(y_test, pred_test))

print("Recall on train data:", recall_score(y_train, pred_train))
print("Recall on test data:", recall_score(y_test, pred_test))

print("Precision on train data:", precision_score(y_train, pred_train))
print("Precision on test data:", precision_score(y_test, pred_test))

print("f1 score on train data:", f1_score(y_train, pred_train))
print("f1 score on test data:", f1_score(y_test, pred_test))


# In[182]:


def make_confusion_matrix(model,library,test_X,y_actual,threshold=0.5,labels=[1, 0]):
    '''
    model : classifier to predict values of X
    library: Takes two arguments stats for statsmodels and sklearn for sklearn library 
    test_X: test set
    y_actual : ground truth  
    threshold: thresold for classifiying the observation as 1
    
    '''
    
    if library == 'sklearn':
        y_predict = model.predict(test_X)
        cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    
        df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
        group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=labels,fmt='')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    elif library =='stats':
        y_predict = model.predict(test_X)>threshold
        cm=metrics.confusion_matrix( y_actual, y_predict, labels=[1,0])
        cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    
        df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
        group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}" for v1, v2 in
                  zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=labels,fmt='')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


# In[183]:


# AUC ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


logit_roc_auc_train = roc_auc_score(y_train, lg2.predict(X_train1))
fpr, tpr, thresholds = roc_curve(y_train, lg2.predict(X_train1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# In[184]:


logit_roc_auc = roc_auc_score(y_test, lg2.predict(X_test1))
fpr, tpr, thresholds = roc_curve(y_test, lg2.predict(X_test1))
plt.figure(figsize=(13, 8))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("Log_ROC")
plt.show()


# * Logistic Regression model is giving a generalized performance on training and test set.
# * ROC-AUC score of 0.90 on training and test set is quite good.

# In[185]:


lg2.summary2()


# ## Coefficient interpretations
# 
# * According to the Coefficients: flight distance, Food and drink, Economy class travelers, Men and age groups 26-35/60-85 are positive and increases in these will lead to increase in chances of the customer being dissatisfied/neutral.
# * Coefficients of Seat comfort, legroom entertainment and Online ease are negative, increase in these will lead to decrease in chances of a passenger being dissatisfied/neutral.

# ## Find Threshold to improve recall using AUC curve

# In[186]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = roc_curve(y_test, lg2.predict(X_test1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(optimal_threshold)


# In[187]:


# Model prediction with optimal threshold
pred_train_opt = (lg.predict(X_train) > optimal_threshold).astype(int)
pred_test_opt = (lg.predict(X_test) > optimal_threshold).astype(int)


# In[188]:


# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = metrics.roc_curve(y_test, lg2.predict(X_test1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)


# In[189]:


def get_metrics_score1(model,train,test,train_y,test_y,threshold=0.5,flag=True,roc=False):
    '''
    Function to calculate different metric scores of the model - Accuracy, Recall, Precision, and F1 score
    model: classifier to predict values of X
    train, test: Independent features
    train_y,test_y: Dependent variable
    threshold: thresold for classifiying the observation as 1
    flag: If the flag is set to True then only the print statements showing different will be displayed. The default value is set to True.
    roc: If the roc is set to True then only roc score will be displayed. The default value is set to False.
    '''
    # defining an empty list to store train and test results
    
    score_list=[] 
    
    pred_train = (model.predict(train)>threshold)
    pred_test = (model.predict(test)>threshold)

    pred_train = np.round(pred_train)
    pred_test = np.round(pred_test)
    
    train_acc = accuracy_score(pred_train,train_y)
    test_acc = accuracy_score(pred_test,test_y)
    
    train_recall = recall_score(train_y,pred_train)
    test_recall = recall_score(test_y,pred_test)
    
    train_precision = precision_score(train_y,pred_train)
    test_precision = precision_score(test_y,pred_test)
    
    train_f1 = f1_score(train_y,pred_train)
    test_f1 = f1_score(test_y,pred_test)
    
    
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,train_f1,test_f1))
        
    
    if flag == True: 
        print("Accuracy on training set : ",accuracy_score(pred_train,train_y))
        print("Accuracy on test set : ",accuracy_score(pred_test,test_y))
        print("Recall on training set : ",recall_score(train_y,pred_train))
        print("Recall on test set : ",recall_score(test_y,pred_test))
        print("Precision on training set : ",precision_score(train_y,pred_train))
        print("Precision on test set : ",precision_score(test_y,pred_test))
        print("F1 on training set : ",f1_score(train_y,pred_train))
        print("F1 on test set : ",f1_score(test_y,pred_test))
   
    if roc == True:
        print("ROC-AUC Score on training set : ",roc_auc_score(train_y,pred_train))
        print("ROC-AUC Score on test set : ",roc_auc_score(test_y,pred_test))
    
    return score_list # returning the list with train and test scores


# In[190]:


optimal_threshold_curve = 0.58

scores_LR = get_metrics_score1(lg2,X_train1,X_test1,y_train,y_test,threshold=optimal_threshold_curve,roc=True)


# ## Improved score summary
# * All metrics scores increased except for Precision, which decreased.
# * Recall greatly improved and there doesn't appear to be any overfitting of the data.

# In[191]:


from sklearn.metrics import precision_recall_curve

y_scores = lg2.predict(X_train1)
prec, rec, tre = precision_recall_curve(
    y_train,
    y_scores,
)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()


# * It looks as if we can decrease our threshold to about 3 without too much of a decrease in precision.

# In[196]:


optimal_threshold_curve = 0.3


# confusion matrix
make_confusion_matrix(lg2, "stats", X_test1, y_test, threshold=optimal_threshold_curve)

# checking model performance
scores_LR = get_metrics_score1(
    lg2, X_train1, X_test1, y_train, y_test, threshold=optimal_threshold_curve, roc=True
)


# * As predicted, recall scores improved and all other scores slightly decreased.
# * We can keep the threshold at 3.

# In[197]:


def make_confusion_matrix(y_actual, y_predict, labels=[1, 0]):
    """
    y_predict: prediction of class
    y_actual : ground truth
    """
    cm = confusion_matrix(y_predict, y_actual, labels=[1, 0])
    df_cm = pd.DataFrame(
        cm, index=[i for i in ["1", "0"]], columns=[i for i in ["1", "0"]]
    )
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(7, 5))
    sns.heatmap(df_cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# In[198]:


make_confusion_matrix(y_test, pred_test_opt)


# # Logistic Regression Summary
# * LG model 2 shows good performance on the training and test sets
# * Model is showing a good balance overall.
# * Area under the curve has decreased as compared to the initial model but the performance is generalized on training and test set.
# * Economy class customers show being the most dissatisfied/neutral customers
# * Men are more likely to be dissatisfied/neutral
# * Passengers over 60 are more dissatisfied/neutral
# * I will try Decision Tree modeling to see if there are more robust models that have more insight
# 
# # Decision Tree

# In[199]:


X = df_join.drop(
    [
        "DepartureDelayin_Mins",
        "Satisfaction",
        "Seat_comfort",
        "depart_arrivalTime.conv",
        "Food_drink",
        "Gate_location",
        "Inflightwifi",
        "Inflight_entertainment",
        "Online_support",
        "Ease_of_Onlinebooking",
        "Onboard_service",
        "Legroom",
        "Baggage_handling",
        "Checkin_service",
        "Cleanliness",
        "Boarding",'Preflight','Inflight'
    ],
    axis=1,
)

y = df_join[["Satisfaction"]]


# In[200]:


# encoding the categorical variables
X = pd.get_dummies(X, columns=["Gender", "Class", "age_bin"], drop_first=True)
X.head()


# In[201]:


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape)


# In[202]:


feature_names = list(X_train.columns)
print(feature_names)


# In[203]:


# Download libraries for Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy.stats import randint

estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(6, 60),
    "max_features": randint(1, 20),
    "min_samples_leaf": randint(1, 20),
    "min_impurity_decrease": [0.001, 1],
    "min_samples_split": [3, 5, 8],
    "criterion": ["gini", "entropy"],
}


# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
rand_obj = RandomizedSearchCV(estimator, parameters, n_iter=50, scoring="recall", cv=10)
rand_obj = rand_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = rand_obj.best_estimator_
estimator.fit(X_train, y_train)


# In[204]:


def make_confusion_matrix(
    model, library, test_X, y_actual, threshold=0.5, labels=[1, 0]
):
    """
    model : classifier to predict values of X
    library: Takes two arguments stats for statsmodels and sklearn for sklearn library
    test_X: test set
    y_actual : ground truth
    threshold: thresold for classifiying the observation as 1

    """

    if library == "sklearn":
        y_predict = model.predict(test_X)
        cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])

        df_cm = pd.DataFrame(
            cm,
            index=[i for i in ["Actual - No", "Actual - Yes"]],
            columns=[i for i in ["Predicted - No", "Predicted - Yes"]],
        )
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = [
            "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
        ]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=labels, fmt="")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    elif library == "stats":
        y_predict = model.predict(test_X) > threshold
        cm = metrics.confusion_matrix(y_actual, y_predict, labels=[1, 0])
        cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0, 1])

        df_cm = pd.DataFrame(
            cm,
            index=[i for i in ["Actual - No", "Actual - Yes"]],
            columns=[i for i in ["Predicted - No", "Predicted - Yes"]],
        )
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = [
            "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
        ]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=labels, fmt="")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")


# In[205]:


def get_metrics_score(
    model, library, train, test, train_y, test_y, threshold=0.5, flag=True, roc=False
):
    """
    Function to calculate different metric scores of the model - Accuracy, Recall, Precision, and F1 score
    library: Takes two arguments stats for statsmodels and sklearn for sklearn library
    model: classifier to predict values of X
    train, test: Independent features
    train_y,test_y: Dependent variable
    threshold: threshold for classifiying the observation as 1
    flag: If the flag is set to True then only the print statements showing different will be displayed. The default value is set to True.
    roc: If the roc is set to True then only roc score will be displayed. The default value is set to False.
    """
    # defining an empty list to store train and test results
    if library == "stats":
        score_list = []

        pred_train = model.predict(train) > threshold
        pred_test = model.predict(test) > threshold

        pred_train = np.round(pred_train)
        pred_test = np.round(pred_test)

        train_acc = accuracy_score(pred_train, train_y)
        test_acc = accuracy_score(pred_test, test_y)

        train_recall = recall_score(train_y, pred_train)
        test_recall = recall_score(test_y, pred_test)

        train_precision = precision_score(train_y, pred_train)
        test_precision = precision_score(test_y, pred_test)

        train_f1 = f1_score(train_y, pred_train)
        test_f1 = f1_score(test_y, pred_test)

        score_list.extend(
            (
                train_acc,
                test_acc,
                train_recall,
                test_recall,
                train_precision,
                test_precision,
                train_f1,
                test_f1,
            )
        )

    elif library == "sklearn":
        score_list = []

        pred_train = model.predict(train)
        pred_test = model.predict(test)

        train_acc = accuracy_score(pred_train, train_y)
        test_acc = accuracy_score(pred_test, test_y)

        train_recall = recall_score(train_y, pred_train)
        test_recall = recall_score(test_y, pred_test)

        train_precision = precision_score(train_y, pred_train)
        test_precision = precision_score(test_y, pred_test)

        train_f1 = f1_score(train_y, pred_train)
        test_f1 = f1_score(test_y, pred_test)

        score_list.extend(
            (
                train_acc,
                test_acc,
                train_recall,
                test_recall,
                train_precision,
                test_precision,
                train_f1,
                test_f1,
            )
        )

    if flag == True:
        print("Accuracy on training set : ", accuracy_score(pred_train, train_y))
        print("Accuracy on test set : ", accuracy_score(pred_test, test_y))
        print("Recall on training set : ", recall_score(train_y, pred_train))
        print("Recall on test set : ", recall_score(test_y, pred_test))
        print("Precision on training set : ", precision_score(train_y, pred_train))
        print("Precision on test set : ", precision_score(test_y, pred_test))
        print("F1 on training set : ", f1_score(train_y, pred_train))
        print("F1 on test set : ", f1_score(test_y, pred_test))

    if roc == True and library == "sklearn":
        pred_train_prob = model.predict_proba(train)[:, 1]
        pred_test_prob = model.predict_proba(test)[:, 1]
        print(
            "ROC-AUC Score on training set : ", roc_auc_score(train_y, pred_train_prob)
        )
        print("ROC-AUC Score on test set : ", roc_auc_score(test_y, pred_test_prob))

    elif roc == True and library == "stats":
        print("ROC-AUC Score on training set : ", roc_auc_score(train_y, pred_train))
        print("ROC-AUC Score on test set : ", roc_auc_score(test_y, pred_test))

    return score_list  # returning the list with train and test scores


# Recall scores are decent, precision a little low. 

# In[206]:


# let us make confusion matrix on train set
make_confusion_matrix(estimator, "sklearn", X_train, y_train)


# In[207]:


plt.figure(figsize=(10, 10))
out = tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[208]:


# Text report showing the rules of a decision tree -

print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# In[209]:


# Choose the type of classifier.
estimator2 = DecisionTreeClassifier(criterion="gini", random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": [6, 15, 25],
    "min_samples_leaf": [7, 10, 14],
    "max_leaf_nodes": [5, 10, 14],
    "min_impurity_decrease": [0.001, 0.01],
    "min_samples_split": [3, 5, 8],
    "max_features": ["log2"],
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(estimator2, parameters, scoring=acc_scorer, cv=10)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator2 = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator2.fit(X_train, y_train)


# In[210]:


make_confusion_matrix(estimator, "sklearn", X_train, y_train)


# In[211]:


get_metrics_score(estimator, "sklearn", X_train, X_test, y_train, y_test)


# In[212]:


make_confusion_matrix(estimator2, "sklearn", X_test, y_test)


# In[213]:


get_metrics_score(estimator2, "sklearn", X_train, X_test, y_train, y_test)


# In[214]:


plt.figure(figsize=(10, 10))
out = tree.plot_tree(
    estimator2,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[215]:


importances = estimator2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ## Observations
# * Randomized tuned tree performed better than the tuned tree with gridsearch
# * Randomsearch tree was still quite large but didn't seem to overfit
# * Inflight entertainment was by far the most important feature, then Seat comfort was next but far behind
# * These models were opposite from the logistic regression model which showed travel class(economy), gender(male) and age bins as significant features to identify disatisfied customers.

# In[216]:


# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# Fitting the model
rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train, y_train)

# Calculating different metrics
get_metrics_score(rf_estimator, "sklearn", X_train, X_test, y_train, y_test)

# Creating confusion matrix
make_confusion_matrix(rf_estimator, "sklearn", X_test, y_test)


# In[217]:


# Choose the type of classifier.
rf_tuned = RandomForestClassifier(random_state=1)

parameters = {
    "max_depth": list(np.arange(3, 10, 1)),
    "max_features": np.arange(0.6, 1.1, 0.1),
    "max_samples": np.arange(0.7, 1.1, 0.1),
    "min_samples_split": np.arange(2, 20, 5),
    "n_estimators": np.arange(30, 160, 20),
    "min_impurity_decrease": [0.0001, 0.001, 0.01, 0.1],
}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
rand_obj = RandomizedSearchCV(
    rf_tuned, parameters, n_iter=50, scoring=scorer, cv=10, n_jobs=-1
)
rand_obj = rand_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rf_tuned = rand_obj.best_estimator_

# Fit the best algorithm to the data.
rf_tuned.fit(X_train, y_train)


# In[218]:


get_metrics_score(rf_tuned, "sklearn", X_train, X_test, y_train, y_test)

# Creating confusion matrix
make_confusion_matrix(rf_tuned, "sklearn", X_test, y_test)


# In[219]:


importances = rf_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# * The decision tree and random forest models are very similar with regard to feature importance related to dissatisfaction.
# * random forest tuned model performed extremely well.
# * Inflight entertainment is still too far ahead of all other features, which is discerning
# 
# ## Bagging Classifier
# * RandomizedSearch Hyperparameter tuning 

# In[220]:


bag_estimator_tuned = BaggingClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_samples": [0.7, 0.8, 0.9, 1],
    "max_features": [0.7, 0.8, 0.9, 1],
    "n_estimators": [10, 20, 30, 40, 50],
}

# Run the grid search
rand_obj = RandomizedSearchCV(bag_estimator_tuned, parameters, scoring=acc_scorer, cv=5)
rand_obj = rand_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
bag_estimator_tuned = rand_obj.best_estimator_

# Fit the best algorithm to the data.
bag_estimator_tuned.fit(X_train, y_train)


# In[221]:


get_metrics_score(bag_estimator_tuned, "sklearn", X_train, X_test, y_train, y_test)

make_confusion_matrix(bag_estimator_tuned, "sklearn", X_test, y_test)


# Bagging estimator is overfitting
# ## Ada Boost Classifer
# * Randomized Search

# In[222]:


get_ipython().run_cell_magic('time', '', '\nabc_tuned = AdaBoostClassifier(random_state=1)\n\n# Grid of parameters to choose from\nparameters = {\n    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),\n                      DecisionTreeClassifier(max_depth=3)],\n    "n_estimators": np.arange(10,110,10),\n    "learning_rate":np.arange(0.1,2,0.1)\n}\n\n\n# Run the grid search\nrand_obj = RandomizedSearchCV(abc_tuned, parameters, scoring=acc_scorer,cv=5)\nrand_obj = rand_obj.fit(X_train, y_train)\n\n# Set the clf to the best combination of parameters\nabc_tuned = rand_obj.best_estimator_\n\n# Fit the best algorithm to the data.\nabc_tuned.fit(X_train, y_train)')


# In[223]:


get_metrics_score(abc_tuned, "sklearn", X_train, X_test, y_train, y_test)

make_confusion_matrix(abc_tuned, "sklearn", X_test, y_test)


# In[224]:


importances = abc_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# * The Ada boost model is overfitting slightly on the training data
# * The model is generalized well overall
# * Seat comfort is by far the most important feature followed by gate location convenience and entertainment
# * There is slightly less of a gap than the decision tree and random forest models when it comes to the most important feature to the second.
# 
# ## XGBoost
# * Randomized Search

# In[225]:


# downloading library
import xgboost as xgb


# In[226]:


get_ipython().run_cell_magic('time', '', '\nxgb_tuned = XGBClassifier(random_state=1, eval_metric=\'logloss\')\n\n# Grid of parameters to choose from\nparameters = {\n    "n_estimators": np.arange(10,100,20),\n    "scale_pos_weight":[0,1,2,5],\n    "subsample":[0.5,0.7,0.9,1],\n    "learning_rate":[0.01,0.1,0.2,0.05],\n    "gamma":[0,1,3],\n    "colsample_bytree":[0.5,0.7,0.9,1],\n    "colsample_bylevel":[0.5,0.7,0.9,1]\n}\n\n# Type of scoring used to compare parameter combinations\nacc_scorer = metrics.make_scorer(metrics.recall_score)\n\n# Run the grid search\ngrid_obj = RandomizedSearchCV(xgb_tuned, parameters,n_iter=60, n_jobs=10, scoring=acc_scorer,cv=5)\ngrid_obj = grid_obj.fit(X_train, y_train)\n\n# Set the clf to the best combination of parameters\nxgb_tuned = grid_obj.best_estimator_\n\n# Fit the best algorithm to the data.\nxgb_tuned.fit(X_train, y_train)')


# In[227]:


get_metrics_score(xgb_tuned, "sklearn", X_train, X_test, y_train, y_test)

make_confusion_matrix(xgb_tuned, "sklearn", X_test, y_test)


# In[228]:


importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[229]:


get_ipython().run_cell_magic('time', '', '\nxgb_tuned2 = XGBClassifier(random_state=1, eval_metric="logloss")\n\n# Grid of parameters to choose from\nparameters = {\n    "n_estimators": np.arange(50, 100, 15),\n    "scale_pos_weight": [2, 5],\n    "subsample": [0.7, 1],\n    "learning_rate": [0.02, 0.05],\n    "gamma": [0],\n    "colsample_bytree": [1],\n    "colsample_bylevel": [0.7],\n}\n\n# Type of scoring used to compare parameter combinations\nacc_scorer = metrics.make_scorer(metrics.recall_score)\n\n# Run the grid search\ngrid_obj = GridSearchCV(xgb_tuned2, parameters, scoring=acc_scorer, cv=5)\ngrid_obj = grid_obj.fit(X_train, y_train)\n\n# Set the clf to the best combination of parameters\nxgb_tuned2 = grid_obj.best_estimator_\n\n# Fit the best algorithm to the data.\nxgb_tuned2.fit(X_train, y_train)')


# In[230]:


get_metrics_score(xgb_tuned2, "sklearn", X_train, X_test, y_train, y_test)

make_confusion_matrix(xgb_tuned2, "sklearn", X_test, y_test)


# In[231]:


importances = xgb_tuned2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[232]:


print(X_train.shape, X_test.shape)


# In[233]:


print(y_train.shape, y_test.shape)


# In[234]:


def metrics_score(model, flag=True):
    """
    model : classifier to predict values of X

    """
    # defining an empty list to store train and test results
    score_list = []

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_recall = metrics.recall_score(y_train, pred_train)
    test_recall = metrics.recall_score(y_test, pred_test)

    train_precision = metrics.precision_score(y_train, pred_train)
    test_precision = metrics.precision_score(y_test, pred_test)

    score_list.extend(
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
        )
    )

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ", model.score(X_train, y_train))
        print("Accuracy on test set : ", model.score(X_test, y_test))
        print("Recall on training set : ", metrics.recall_score(y_train, pred_train))
        print("Recall on test set : ", metrics.recall_score(y_test, pred_test))
        print(
            "Precision on training set : ", metrics.precision_score(y_train, pred_train)
        )
        print("Precision on test set : ", metrics.precision_score(y_test, pred_test))

    return score_list  # returning the list with train and test scores


# In[235]:


# defining list of models
models = [estimator, rf_tuned, bag_estimator_tuned, abc_tuned, xgb_tuned]

# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []

# looping through all the models to get the metrics score - Accuracy, Recall and Precision
for model in models:

    j = metrics_score(model, False)
    acc_train.append(np.round(j[0], 2))
    acc_test.append(np.round(j[1], 2))
    recall_train.append(np.round(j[2], 2))
    recall_test.append(np.round(j[3], 2))
    precision_train.append(np.round(j[4], 2))
    precision_test.append(np.round(j[5], 2))


# In[236]:


comparison_frame = pd.DataFrame(
    {
        "Model": [
            "Decision Tree Tuned",
            "Random Forest Tuned",
            "Bagging Classifier",
            "Ada Boost Tuned",
            "XGBoost Tuned",
        ],
        "Train_Accuracy": acc_train,
        "Test_Accuracy": acc_test,
        "Train_Recall": recall_train,
        "Test_Recall": recall_test,
        "Train_Precision": precision_train,
        "Test_Precision": precision_test,
    }
)
comparison_frame


# In[237]:


# Created a separate data output frame for logistic regression because the x_train shapes were different from the tree models
comparison_frame2 = pd.DataFrame(
    {
        "model": ["lg2"],
        "Train_Accuracy": accuracy_score(y_train, pred_train),
        "Test_Accuracy": accuracy_score(y_test, pred_test),
        "Train_Recall": recall_score(y_train, pred_train),
        "Test_Recall": recall_score(y_test, pred_test),
        "Train_Precision": precision_score(y_train, pred_train),
        "Test_Precision": precision_score(y_train, pred_train),
    }
)
comparison_frame2


# ## Conclusion
# * All models performed fairly well, including the logistic regression models
# * Ada Boost model performed great and generalized best, similar to the random forest model
# * Xgboost has the highest recall with other scores still in a good range.
# * Even though xgboost has the highest recall I feel that the Adaboost model is the best model to report on because of the lesser gap in feature importance, and also being the model that generalized best 
# * Seat comfort is the most important feature, followed by gate location convenience, inflight entertainment, food and drink and leg room.

# ## Summary
# * Inflight entertainment was among the top important features for all models but showed as number 3 feature on the ADA boost and showed an as an important feature mostly to the 7 to 18 age group in the EDA. 
# * Seat comfort was the most important feature in our ada boost model and did notice it as very important in passengers 65 and up age range in the EDA, but apparently was still important across all age groups
# * Gate location didn't stand out in the EDA but was apparently an important feature to most models but especially the Ada model.
# * Economy class dissatisfaction was apparent in the EDA and was a significant feature for all models, but in the mid range of importance for the Ada model. Therefore feel that it should still be considered among the top 5 features overall.
# * Though there is preferred model its still important to group the discoveries from the EDA and other good performing models and blend the information for recommendations.  And in this case keeping the Eco class, inflight wifi and online ease among the top important features along with seat comfort, inflight wifi and gate location.
# 
# 

# # Business Recommendations
# * The analysis shows that seat comfort and leg room are big factors that contribute to customer dissatisfaction. Though still present in business class, the EDA shows that it is more prominent in economy class.  The company needs to work with partners to restructure their seat designs and possibly find a way to make a little more legroom at the same time. 
#     -First they need to make sure the new designs are part of every new plane being built.  
#     -Then work on their current fleet of airplanes at a rate that makes sense financially and rebuild        -          the seating plane by plane. 
# * Inflight entertainment is another important feature that needs to be immediately addressed.  Similar to what Easyjet and Ryanair are currently doing with Panasonic, Falcon airlines can find an electronics and/or entertainment company to brainstorm different ways to revamp their entire inflight entertainment and follow through.  
# * Food and drink are also showed up in analysis as needing attention.  The airline can conduct another passenger survey to ask what types of food and beverage items they would like to see offered.  Do more research and investigate what other airlines and even hotels are offering, then work with their current vendors or find new ones to accomplish their discovered goals.
# * Though not apparent in the EDA, gate location showed up as important in the model.  The airline can work with airports, especially their hub airports, to try and relocate to a better, more convenient area.
# * All in all, if the airline works on improving a few important areas at a time, they will get happier customers. If the comfort and amenties are present, along with the right pricing, they could become less annoyed by the other disadvantages. So start small, conduct another sat survey and work their way through the next set of hurdles.
# 
