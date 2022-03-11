#!/usr/bin/env python
# coding: utf-8

# ## Problem

# Even thought there are many other factors when it comes to purchasing a new car, like warranty and vehicle history, accidents, etc. But basic data can help produce pricing models.
# 
# * Does various predicting factors really affect the price?
# * What are the predicting variables actually affecting price?
# * How much does location have to do with pricing, can they sell the same car/mileage etc in Mumbai for more or less than in Chennai?
# * Are higher end models important in the used market?
# * How does kilometers driven affect price?
# * How much of an impact does transmission and fuel type have on kilometers driven and pricing?
# * Do engine, Power matter when it comes to pricing?
# * How important are make and model when it comes to pricing?  Can you charge slightly more for vehicles when there are not that many available? 
# 
# ## Assumptions
# 
# The used car data is a simple random sample from the population data.

# ## Loading Libraries

# In[1]:


import numpy as np  #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set (color_codes=True)
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import warnings


# In[2]:


pd.set_option(
    "display.max_columns", None
)  # this is so I don't have a limit on displayed rows and columns when
# pd.set_option('display.max_rows', None)  # I print
pd.set_option("display.max_rows", None)


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


get_ipython().run_line_magic('reload_ext', 'nb_black')


# ## Loading and exploring data

# In[5]:


cars = pd.read_csv("used_cars_data.csv")  # downloading the data file
cars.head()


# In[6]:


print(f"There are {cars.shape[0]} rows and {cars.shape[1]} columns")
np.random.seed(
    31
)  # Using a random seed so I can use throughout my queries and get the same random results
cars.sample(n=100)  # Looking at a different set of random rows from first 5


# In[7]:


cars.shape


# In[8]:


cars.info()


# ## Observations
# * There are 7253 rows and 12 columns
# * Are are 9 columns that are objects.  Some need to be converted to floats by removing units, and some need to be change to categorical
# * Price and New Price have the highest number of null values
# 

# In[9]:


cars.drop(
    ["S.No."], axis=1, inplace=True
)  # after looking at the data, it doesn't appear that Serial number wil
# be usefull in our analysis, therefore dropping the column


# In[10]:


cars.info()  # checking to see if Serial number column is dropped


# In[11]:


cars.isnull().sum().sort_values(
    ascending=False
)  # checking the number of missing values for each column


# In[12]:


cars.Transmission.value_counts()


# In[13]:


cars.Fuel_Type.value_counts()


# In[14]:


cars.Seats.unique()


# In[15]:


cars.Location.unique()


# In[16]:


cars.Owner_Type.value_counts()


# In[17]:


pd.set_option(
    "display.float_format", lambda x: "%.3f" % x
)  # to display numbers in digits
cars.describe(include="all").T


# In[18]:


cars["Power"].value_counts()


# In[19]:


km_max = cars[cars["Kilometers_Driven"] == 6500000]
print(km_max)


# ## Observation
# * A fair number of columns do not have missing values
# * Mileage is only missing 2, so out of 7253 total we can probably drop those two rows
# * New price and price have the most number of missing values.  We can look up those vehicles and find same year, make/model/Trim and input the new price values.  Price values will take more work as we will have to find comparable kilometers driven and number owners.
# * Power and engine have the exact same number of missing numbers.  Engine displacement has a direct impact on  power so are somewhat tied together. We may only need one of those columns but will see with further EDA. 
# * My theory is that the missing values are tied to electric vehicles.

# ## Processing Columns
# Before I go any further I have to turn some columns into numerical data by simply getting rid of strings and the objects need to be turned into categorical.
# 

# In[20]:


price_cols = []
for colname in cars.columns[cars.dtypes == "object"]:  # find only string columns
    if (
        cars[colname].str.endswith("Lakh").any()
    ):  # using `.str` so I can use an element-wise string method
        price_cols.append(colname)
print(price_cols)


# In[21]:


cars["New_Price"].sample(25)


# In[22]:


price_cols = []
for colname in cars.columns[cars.dtypes == "object"]:  # find only string columns
    if (
        cars[colname].str.endswith("Lakh").any()
    ):  # using `.str` so I can use an element-wise string method
        price_cols.append(colname)
print(price_cols)


# * This shows me that only new price contains the currency string, not price.  So now I remove Lakh from new price column to get a numeric value.
# * I will also convert the Lakh unit to Rupees.
# * I do this first by creating a new column for New Price and iterate through New Price finding at least one of the currencies I need to change and then assign it to the new column (above)
# * I define a function to convert the currency to Lakh and remove the all currency names
# * Then I ended up removing the column after further analysis
# 

# In[23]:


def same_curr(New_Price):
    if isinstance(New_Price, str):
        if New_Price.endswith("Lakh"):
            multiplier = 1
        elif New_Price.endswith("Cr"):
            multiplier = 100
        return float(New_Price.replace("Lakh", "").replace("Cr", "")) * multiplier
    else:
        return np.nan


for colname in price_cols:
    cars[colname] = cars[colname].apply(same_curr)

cars[price_cols].sample(10)


# In[24]:


cars.New_Price.dtype


# * I have changed New Price to a float and converted Cr into Lakh, so now all the data in that column can be used.
# 
# * Now its time to remove the units of measurements from the rest of columns that can be changed to numerical columns and convert them from objects to floats. 
# 
# * I have done the same for Engine column as I have for New Price, no other conversions were necessary.

# In[25]:


eng_col = []
for colname in cars.columns[cars.dtypes == "object"]:
    if cars[colname].str.endswith("CC").any():
        eng_col.append(colname)
print(eng_col)


# In[26]:


def cc_to_num(Engine):
    if isinstance(Engine, str):
        if Engine.endswith("CC"):
            return float(Engine.replace("CC", ""))
    else:
        return np.nan


for colname in eng_col:
    cars[colname] = cars[colname].apply(cc_to_num)

cars[eng_col].head()


# In[27]:


cars.Engine.dtype


# In[28]:


cars.Mileage = (
    cars.Mileage.str.replace("km/kg", "").str.replace("kmpl", "").str.replace("", "")
)


# In[29]:


cars["Mileage"] = cars["Mileage"].astype("float")

cars.Mileage.head()
# I have removed all of the units of measurement string and now need to convert Power and Mileage from object to floats.

# In[30]:


cars["Mileage"].dtype


# In[31]:


cars["Power"].isnull().sum()


# In[32]:


cars.Power = cars.Power.str.replace(" bhp", "").str.replace("null", "")
cars.Power.head()


# In[33]:


cars["Power"] = cars["Power"].apply(lambda x: np.nan if x == "" else x)
cars["Power"].isnull().sum()


# In[34]:


cars["Power"] = cars["Power"].astype("float")
cars.Power.head()


# In[35]:


cars.Power.value_counts()


# Obersvation: There were a number of misc strings and empty space and white spaces in this column

# ## Summary
# All data that should have been numeric but were objects have now been processed and converted to float data types. 
# Now we can revisit summary statistics and then start looking more into objects, and if possible, replace the columns with summaries and converting them into categories.

# In[36]:


cars.describe()


# ## New observations on statiscally summary
# * There is still a lot of missing data for a few key columns which we will address later
# * oldest car is 1996, newest is 2019
# * Mean and Median are the same for Mileage
# * Seems like Engine have outliers on both ends of the spectrum, as does power, but since those two variables are highly connected from an engineering perspective, its not surprising
# * Mean and median for price and new price are very different.  I hypothesize that it has a lot to do with the large number of missing values
# * Most of the cars for sale have 5 seats as that numbers is shown for all 3 quartiles.  The zero min most likely respresents missing values and the max are probably vans or busses.

# In[37]:


num_missing = cars.isnull().sum(axis=1)
num_missing.value_counts()


# In[38]:


cars.drop(
    ["New_Price"], axis=1, inplace=True
)  ### I decided to drop the New Price column as there were too many
### missing values and too many makes and models to try and match up


# In[39]:


num_missing = cars.isnull().sum(axis=1)
num_missing.value_counts()


# In[40]:


cars.isnull().sum().sort_values(ascending=False)


# In[41]:


for n in (
    num_missing.value_counts().sort_index().index
):  ## I ran this cell to show exactly what rows are missing values
    if n > 0:
        print(f"For the rows with exactly {n} missing values, NAs are found in:")
        n_miss_per_col = cars[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print("\n\n")


# In[42]:


cars[num_missing == 2].sample(n=5)


# ## Visuals of numerical data
# Defining and setting up plots to plug in numerical data and visual assess what we are working with to produce our models

# In[43]:


def histogram_boxplot(feature, figsize=(15, 10), bins=None):

    sns.set(font_scale=2)
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color="red")
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins) if bins else sns.distplot(
        feature, kde=False, ax=ax_hist2
    )
    ax_hist2.axvline(feature.mean(), color="g", linestyle="--")
    ax_hist2.axvline(feature.median(), color="black", linestyle="-")


# In[44]:


histogram_boxplot(cars.Price)


# Price is heavily right skewed.
# Mean is 9.5 and median is about half that at 5.6.
# There appears to be a large number of outliers and missing values that need to be addressed.

# In[45]:


histogram_boxplot(cars.Year)


# Even though this graph looks left skewed, I feel its close to normal standardization.  Have a few older cars as outliers is probably to blame and will will do some diagnostics to reconcile.

# In[46]:


histogram_boxplot(cars.Kilometers_Driven)


# Kilometers Driven is heavily right skewed. My hypothesis is because there is one astronomical data point throwing the entire distribution off.  It is highly unlikely that a 4 year old car can drive 6500000 kilometers, let alone a 44 year old car.

# In[47]:


sns.pairplot(data=cars)  ## side by side visualization of possible correlations
plt.show()


# ## Observation
# * There appears to be a high correlation between price and year
# * There is also a good correlation between Power and Engine and Price
# * Mileage and price appear to have a negative correlation
# * Kilometers driven is a big factor with price but its hard to tell with its skewness.
# 
# ## Bivariate Analysis
# * A deeper look at correlations

# In[48]:


numeric_columns = cars.select_dtypes(
    include=np.number
).columns.tolist()  # dropping year column as it is temporal variable
corr = (
    cars[numeric_columns].corr().sort_values(by=["Price"], ascending=False)
)  # sorting correlations w.r.t life expectancy

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(28, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    cmap="seismic",
    annot=True,
    fmt=".1f",
    vmin=-1,
    vmax=1,
    center=0,
    square=False,
    linewidths=0.7,
    cbar_kws={"shrink": 0.5},
)


# * Strong positive correlations here are Engine and Power and Price.
# * Strong negative correlations are Mileage, Engine/Price
# * The map shows connection between price and year and mileage and year.  
# * We may be able to remove either power or engine for cleaner data.
# 
# Going to take a closer look at the highly correlated data

# In[49]:


plt.figure(figsize=(10, 7))
sns.scatterplot(y="Price", x="Power", hue="Fuel_Type", data=cars)


# The more power, the higher the price.  From the graph, it looks like most are diesel.

# In[50]:


plt.figure(figsize=(10, 7))
sns.scatterplot(y="Price", x="Engine", hue="Fuel_Type", data=cars)


# As we suspected, engine shows an extremely similar output to Power.  higher engine cc increases in price, and again, majority seems to be diesel. 

# In[51]:


plt.figure(figsize=(10, 7))
sns.lineplot(y="Price", x="Year", data=cars)


# Overall, price increases drastically as the year of the car increases.
# 
# Now we are going to take some of the skewness out of the data. 
# 
# I first want to remove all of the remaining missing values by using a median filler function.

# In[52]:


medianFiller = lambda x: x.fillna(x.median())
numeric_columns = cars.select_dtypes(include=np.number).columns.tolist()
cars[numeric_columns] = cars[numeric_columns].apply(
    medianFiller, axis=0
)  # we will replace missing values in every column with its median


# ## We need to convert the remaining objects to category to continue with our EDA

# In[53]:


cars["Name"] = cars["Name"].astype("category")
cars["Location"] = cars["Location"].astype("category")
cars["Fuel_Type"] = cars["Fuel_Type"].astype("category")
cars["Transmission"] = cars["Transmission"].astype("category")
cars["Owner_Type"] = cars["Owner_Type"].astype("category")


# In[54]:


cars.Location.value_counts()


# In[55]:


plt.figure(figsize=(17, 9))
sns.boxplot(y="Location", x="Price", data=cars, hue="Transmission")
plt.show()


# ## Observation
# * Automatic appears more than manual and manual shows a large number of outliers
# * Though Mumbai is the most frequent location in this dataset, its median rests at a lower price and shows the most outliers in the higher price
# * Coimbatore and Bangalore seemed to have more of an even distribution
# * Just going by this visual, it appears that location has some affect on pricing strategy

# ## Outliers treatment and model scaling
# * We will look at the outliers in each numeric column and treat them
# * We will then scale and plot the variables that were skewed because of the outliers or otherwise

# In[56]:


plt.figure(figsize=(20, 30))

for i, variable in enumerate(numeric_columns):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(cars[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# * All columns have outliers
# * Year has all lower outliers showing that there are less older models on the market
# * Power, price and Engine all have a high number of upper outliers
# * Seats may not be an important column as most sit at the 5 marker
# * Kilometers driven needs to be addressed
# 
# We will now define and treat the outliers.

# In[57]:


def treat_outliers(cars, col):

    Q1 = cars[col].quantile(0.25)  # 25th quantile
    Q3 = cars[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR

    cars[col] = np.clip(cars[col], Lower_Whisker, Upper_Whisker)

    return cars


def treat_outliers_all(cars, col_list):
    """
    treat outlier in all numerical variables
    col_list: list of numerical variables
    df: data frame
    """
    for c in col_list:
        cars = treat_outliers(cars, c)

    return cars


# In[58]:


numerical_col = cars.select_dtypes(include=np.number).columns.tolist()
cars = treat_outliers_all(cars, numerical_col)


# In[59]:


plt.figure(figsize=(20, 30))
for i, variable in enumerate(numeric_columns):
    plt.subplot(5, 4, i + 1)
    plt.boxplot(cars[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)


# In[60]:


cols_to_log = [
    "Price"
]  ### using the following functions to help smooth out the skewed data and normalize
for colname in cols_to_log:
    plt.hist(cars[colname], bins=50)
    plt.title(colname)
    plt.show()
    print(np.sum(cars[colname] <= 0))


# In[61]:


cols_to_log = ["Kilometers_Driven"]
for colname in cols_to_log:
    plt.hist(cars[colname], bins=50)
    plt.title(colname)
    plt.show()
    print(np.sum(cars[colname] <= 0))


# Log doesn't work for price as well as Kilometers Driven after outlier treatment.  
# It looks like there were a few entered as a mistake.  Lets other methods of scaling.

# In[62]:


plt.hist(
    np.log(cars["Price"] + 1), 10
)  ## Data that has a 0 or negative value doesn't show up with log so we use
plt.title("log(Price + 1)")  # these functions
plt.show()
plt.hist(np.arcsinh(cars["Price"]), 10)
plt.title("arcsinh(Price)")
plt.show()
plt.hist(np.sqrt(cars["Price"]), 10)
plt.title("sqrt(Price)")
plt.show()


# ## Categorical Variables
# 
# It would be great if we separate make and model, but there would be many dummy variables and things would get messy and probably wouldn't yield the best statistical outputs.
# 
# I am going to separate make from model to at least group car brands together. We may or may not drop the model column in the end.

# In[63]:


cars.Name.duplicated().sum()  # seeing how many duplicates there are


# In[64]:


cars.Name.describe()  ### There are no missing values but 2041 unique values.  We need to split by make and model
### so we can predict fill in price for missing values


# In[65]:


cars.Name.value_counts(
    ascending=True
)  ## There are a large number of various makes and models


# In[66]:


cars.sort_values(by="Name").head()


# In[67]:


car_make = cars["Name"].str.split(" ", n=1, expand=True)
car_make.head()


# In[68]:


cars["Make"] = car_make[0]
cars["Model"] = car_make[1]
cars.sample(10)


# cars.Make.value_counts(ascending=True)

# In[69]:


cars.head()


# In[70]:


cars.Model.describe()


# ## Observation
# * There are 33 Makes and 2041 different models
# * Mahindra XUV500 W8 2WD is the model that seems to be available the most
# * Maruti shows to be a widely available make
# * There are a small number of makes/brands that that have very little data to contribute 

# In[71]:


num_to_display = 33  # defining this up here so it's easy to change later if I want
for colname in cars.dtypes[cars.dtypes == "object"].index:
    val_counts = cars[colname].value_counts(dropna=False)  # i want to see NA counts
    print(val_counts[:num_to_display])
    if len(val_counts) > num_to_display:
        print(f"Only displaying first {num_to_display} of {len(val_counts)} values.")
    print("\n\n")  # just for more space between


# In[72]:


avg_price = (
    cars.groupby(["Make"])["Price"].mean().sort_values(ascending=False).reset_index()
)
avg_price


# ## Observations
# * The two car makes that represent 38% of the data are Maruti and Hyundai
# * Lamborghini and Ambassador only have one value at opposite ends of the price spectrum.  These two could possibly be dropped.
# * There are two Isuzu rows that can be merged, even though their total count is only 5, after merge
# * Highest average prices are from higher end vehicles, except that Jeep and Mini are ahead of Mercedes-Benz.  That could be because of year or owners
# 
# Before I start encoding all the categories, I want to define my x and y variables for modeling

# In[73]:


X = cars.drop(["Price", "Model", "Name"], axis=1)
y = cars[["Price"]]

print(X.head())
print(y.head())


# In[74]:


print(X.shape)
print(y.shape)


# Now to encode our categorical variables

# In[75]:


X = pd.get_dummies(
    X,
    columns=[
        "Location",
        "Make",
        "Owner_Type",
        "Fuel_Type",
        "Transmission",
    ],
    drop_first=True,
)
X.head()


# In[76]:


### I am splitting the training and test data
### my random state from earlier was 31 so I will stick with that number
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=31
)


# Next in this order I am going to fit the model on the train data at 70% and check the coeffients and intercept

# In[77]:


from sklearn.linear_model import LinearRegression

linearregression = LinearRegression()
linearregression.fit(X_train, y_train)


# In[78]:


coef_df = pd.DataFrame(
    np.append(linearregression.coef_[0], linearregression.intercept_[0]),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)

coef_dfcoef_df = pd.DataFrame(
    np.append(linearregression.coef_[0], linearregression.intercept_[0]),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)

coef_df


# Now I need to check the performance of the model using different (MAE, MAPE, RMSE, R2).
# 
# 
# It seems like I have too much data and may need to eliminate the model column

# In[79]:


intercept = linearregression.intercept_[0]
print("The intercept for our model is {}".format(intercept))


# In[80]:


linearregression.score(X_train, y_train)


# In[81]:


linearregression.score(X_test, y_test)


# In[82]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[83]:


def mape(targets, predictions):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# defining common function for all metrics
def model_perf(model, inp, out):
    """
    model: model
    inp: independent variables
    out: dependent variable
    """
    y_pred = model.predict(inp).flatten()
    y_act = out.values.flatten()

    return pd.DataFrame(
        {
            "MAE": mean_absolute_error(y_act, y_pred),
            "MAPE": mape(y_act, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_act, y_pred)),
            "R^2": r2_score(y_act, y_pred),
        },
        index=[0],
    )


# In[84]:


print("Train Performance\n")
model_perf(linearregression, X_train, y_train)


# In[85]:


print("Test Performance\n")
model_perf(linearregression, X_test, y_test)


# ## Observation
# * My first run training and test sets were not close at 86% and 72%.  It is overfitting a bit.
# * The actual prediction in price isn't so bad at 1.4 Lakh, but the percentage prediction within 25% is not good.  I need to clean up some more data.  I started with dropping the model column.
# * The second set I ran with the model column overfit even more, so I needed to investigate more.

# In[86]:


quartiles = np.quantile(
    cars["Price"][cars["Price"].notnull()], [0.25, 0.75]
)  ## treating/removing extreme outliers
power_4iqr = 4 * (quartiles[1] - quartiles[0])
print(f"Q1 = {quartiles[0]}, Q3 = {quartiles[1]}, 4*IQR = {power_4iqr}")
outlier_powers = cars.loc[
    np.abs(cars["Price"] - cars["Price"].median()) > power_4iqr, "Price"
]
outlier_powers


# I've attempted reworking the data, by removing extreme outliers, deleting columns and binning.  It is all staying steady at these numbers.  The only difference from before was that I was overfitting, now they are about the same.

# In[87]:


import statsmodels.api as sm


# In[88]:


X = sm.add_constant(X, has_constant="add")
X_train1, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=31
)


# In[89]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[90]:


olsmod0 = sm.OLS(y_train, X_train1)
olsres0 = olsmod0.fit()
print(olsres0.summary())


# In[ ]:





# In[91]:


vif_1 = pd.Series(
    [variance_inflation_factor(X_train1.values, i) for i in range(X_train1.shape[1])],
    index=X.columns,
)
print(vif_1)


# In[92]:


X_train2 = X_train1.drop("Seats", axis=1)
vif2 = pd.Series(
    [variance_inflation_factor(X_train2.values, i) for i in range(X_train2.shape[1])],
    index=X_train2.columns,
)
print("VIF Scores: \n\n{}\n".format(vif2))


# Dropping Seats and Years seems to have lowered the VIF values, but they are still significantly higher than 5.  I will check the values again.

# In[93]:


olsmod1 = sm.OLS(y_train, X_train2)
olsres1 = olsmod1.fit()
print(olsres1.summary())


# In[94]:


X_train2.columns


# In[95]:


X_train3 = X_train2.drop(
    [
        "Make_Audi",
        "Make_BMW",
        "Make_Bentley",
        "Make_Chevrolet",
        "Make_Datsun",
        "Make_Fiat",
        "Make_Force",
        "Make_Ford",
        "Make_Hindustan",
        "Make_Honda",
        "Make_Hyundai",
        "Make_ISUZU",
        "Make_Isuzu",
        "Make_Jaguar",
        "Make_Jeep",
        "Make_Lamborghini",
        "Make_Land",
        "Make_Mahindra",
        "Make_Maruti",
        "Make_Mercedes-Benz",
        "Make_Mini",
        "Make_Mitsubishi",
        "Make_Nissan",
        "Make_OpelCorsa",
        "Make_Porsche",
        "Make_Renault",
        "Make_Skoda",
        "Make_Smart",
        "Make_Tata",
        "Make_Toyota",
        "Make_Volkswagen",
        "Make_Volvo",
    ],
    axis=1,
)

vif3 = pd.Series(
    [variance_inflation_factor(X_train3.values, i) for i in range(X_train3.shape[1])],
    index=X_train3.columns,
)
print("VIF Scores: \n\n{}\n".format(vif3))


# In[96]:


olsmod2 = sm.OLS(y_train, X_train3)
olsres2 = olsmod2.fit()
print(olsres2.summary())


# VIF values went down considerably, but so did the R-sq values
# There is one more group I can try and get remove to see what that does to the model

# In[97]:


X_train4 = X_train3.drop(
    ["Fuel_Type_Diesel", "Fuel_Type_Electric", "Fuel_Type_LPG", "Fuel_Type_Petrol"],
    axis=1,
)

vif4 = pd.Series(
    [variance_inflation_factor(X_train4.values, i) for i in range(X_train4.shape[1])],
    index=X_train4.columns,
)
print("VIF Scores: \n\n{}\n".format(vif4))


# We knew there was correlation between Power and Engine from the beginning but didn't know how much.  It seems to be the only thing that is preventing all numbers to be below 5.  We will drop Power as well.

# In[98]:


olsmod3 = sm.OLS(y_train, X_train4)
olsres3 = olsmod3.fit()
print(olsres3.summary())


# In[99]:


X_train5 = X_train4.drop("Engine", axis=1)
vif5 = pd.Series(
    [variance_inflation_factor(X_train5.values, i) for i in range(X_train5.shape[1])],
    index=X_train5.columns,
)
print("VIF Scores: \n\n{}\n".format(vif5))


# In[100]:


olsmod4 = sm.OLS(y_train, X_train5)
olsres4 = olsmod4.fit()
print(olsres4.summary())


# Dropping location got all of the VIF values under 5, however there were still some high P-values and the R-squared was still low.  I will continue to see if we can get it closer.

# In[101]:


X_train6 = X_train5.drop(
    ["Owner_Type_Fourth & Above", "Owner_Type_Second", "Owner_Type_Third"],
    axis=1,
)


vif6 = pd.Series(
    [variance_inflation_factor(X_train6.values, i) for i in range(X_train6.shape[1])],
    index=X_train6.columns,
)
print("VIF Scores: \n\n{}\n".format(vif6))


# In[102]:


olsmod5 = sm.OLS(y_train, X_train6)
olsres5 = olsmod5.fit()
print(olsres5.summary())


# I removed the remainimg columns that I thought posed a problem hower the R-square didn't change and stayed at 64.  I added Location and owner type back in and the R-square values went up though the p-values are high for some categories.

# In[103]:


X_train7 = X_train6.drop(
    [
        "Location_Bangalore",
        "Location_Chennai",
        "Location_Coimbatore",
        "Location_Delhi",
        "Location_Hyderabad",
        "Location_Jaipur",
        "Location_Kochi",
        "Location_Kolkata",
        "Location_Mumbai",
        "Location_Pune",
    ],
    axis=1,
)

vif7 = pd.Series(
    [variance_inflation_factor(X_train7.values, i) for i in range(X_train7.shape[1])],
    index=X_train7.columns,
)
print("VIF Scores: \n\n{}\n".format(vif7))


# In[104]:


olsmod6 = sm.OLS(y_train, X_train7)
olsres6 = olsmod6.fit()
print(olsres6.summary())


# In[105]:


residual = olsres6.resid
np.mean(residual)


# Mean residual is very close to 0, now to find the predicted values.

# In[106]:


residual = olsres5.resid
fitted = olsres5.fittedvalues


# In[107]:


sns.set_style("whitegrid")
sns.residplot(fitted, residual, color="blue", lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted plot")
plt.show()


# Residuals are very low and low p-values but there is still a pattern

# In[108]:


sns.distplot(residual)
plt.title("Normality of residuals")
plt.show()


# The QQ plot of residuals can be used to visually check the normality assumption. The normal probability plot of residuals should approximately follow a straight line.

# In[109]:


import pylab
import scipy.stats as stats

stats.probplot(residual, dist="norm", plot=pylab)
plt.show()


# In[110]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(residual, X_train7)
lzip(name, test)


# Since p-value > 0.05, we can say that the residuals are homoscedastic. This assumption is therefore valid in the data.
# 
# Now we have checked all the assumptions and they are satisfied, so we can move towards the prediction part.

# In[111]:


X_train7.columns


# In[112]:


X_test_final = X_test[X_train7.columns]


# In[113]:


X_test_final.head()


# In[114]:


print("Train Performance\n")
model_perf(olsres6, X_train7.values, y_train)


# * The model has low test and train RMSE and MAE, and both the errors are comparable. So, our model is not overfitting.
# * The model is able to explain 66% of the variation on the test set
# * The MAPE on the test set suggests we can predict within 34% of price.

# In[115]:


olsmod6 = sm.OLS(y_train, X_train7)
olsres6 = olsmod6.fit()
print(olsres6.summary())


# In[ ]:




