#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this will help in making the Python code more structured automatically (good coding practice)
get_ipython().run_line_magic('load_ext', 'nb_black')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist

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


# In[2]:


# To suppress the warning
import warnings

warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_excel("Credit_Card_Data.xlsx")


# In[4]:


# Showing a sample of rows instead of head or tail
print(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

np.random.seed(1)
df.sample(n=50)


# In[5]:


df.info()


# In[6]:


df.nunique()


# In[7]:


# dropping the columns not useful for analysis
df.drop(["Sl_No", "Customer Key"], axis=1, inplace=True)


# In[8]:


df.isnull().sum()


# * There are no null values

# In[9]:


# Looking for duplicates
df.duplicated().sum()


# In[10]:


# There are 11 duplicates, we will remove them from the dataset
df = df[(~df.duplicated())].copy()


# In[11]:


# copying the data to another variable to avoid any changes to original data
data = df.copy()


# In[12]:


data.describe()


# * The duplicated rows are gone and now have 649 rows and 5 columns
# * There doesn't appear to be anything out of the ordinary or problematic other than skewness
# * There are zero values but those are important for anaylsis as there are some customers who never visit the bank, call the bank or do online banking.
# * Lets see how many 0 values there are and if there are any patterns

# In[13]:


(data.Total_visits_bank == 0).sum()


# In[14]:


(data.Total_visits_online == 0).sum()


# In[15]:


(data.Total_calls_made == 0).sum()


# In[16]:


pd.set_option("display.max_rows", 100)


# In[17]:


data[(data.Total_visits_bank == 0)]


# In[18]:


pd.set_option("display.max_rows", 141)
data[(data.Total_visits_online == 0)]


# From the sample of columns above, visits to bank, online and calls made all have varying values.
# This implies to me that people have a pattern of choosing how they prefer to bank, people who don't like to visit the bank primarily bank online and call.

# ## EDA: Univariate

# In[19]:


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
        feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(
        feature, kde=F, ax=ax_hist2, bins=bins, palette="blue"
    ) if bins else sns.distplot(
        feature, kde=False, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        np.mean(feature), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        np.median(feature), color="black", linestyle="-"
    )  # Add median to the histogram


# In[20]:


histogram_boxplot(data["Avg_Credit_Limit"])


# * Average credit limit is heavily right skewed.
# * There is a greater amount of customers that have a credit limit under $18000
# * There are a large number of outliers having very high credit limits

# In[21]:


histogram_boxplot(data["Total_Credit_Cards"])


# ##### * Total credit cards seem to have somewhat of a normal distribution
# * Most customers have between 4-6 cards

# In[22]:


histogram_boxplot(data["Total_visits_bank"])


# In[23]:


histogram_boxplot(data["Total_visits_online"])


# In[24]:


histogram_boxplot(data["Total_calls_made"])


# * Total bank visits has a close to normal distribution
# * Total visits online and calls made are both right skewed, mean and mediansfor both between 2 and 3.
# ## Bivariate Analysis

# In[25]:


# selecting numerical columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()


# In[26]:


# pairplot to see the first set of correlations
sns.pairplot(data)


# In[27]:


# heatmap to see numerical correlation values
plt.figure(figsize=(15, 7))
sns.heatmap(
    data[num_cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="plasma"
)
plt.show()


# * There are very high correlations with average credit limit and total credit cards and visits online:  The higher the credit limit the more cards someone has
# * There is a good correlation with bank visits and total credit cards: Customers with more credit cards tend to visit the bank a little more.
# * There is a high negative correlation with total bank visits and calls made and visits online: As noticed in the summary and histogram boxplots, customers who visit the bank more tend to go online and call less.
# * There is a very high negative correlation with total credit cards and total calls made:  Customer with higher amounts of credit cards call less.
# * There is also a substantial negative correlation between average credit limit and total calls made: Like the above observation, same goes for calls, which makes sense since there is a high correlation with credit limit and credit cards.
# 
# ## We must scale the data before we begin clustering
# Standardizing data to bring each column to a mean of 0 and standard deviation of 1

# In[28]:


# Scaling the data set before clustering
scaler = StandardScaler()
subset = df[num_cols].copy()
subset_scaled = scaler.fit_transform(subset)


# In[29]:


# Creating a dataframe from the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)


# In[30]:


subset_scaled_df.head()


# In[31]:


# defining k-means clusters
clusters = range(1, 9)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction = model.predict(subset_scaled_df)
    distortion = (
        sum(
            np.min(cdist(subset_scaled_df, model.cluster_centers_, "euclidean"), axis=1)
        )
        / subset_scaled_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)


# * Appropriate k using elbow method looks to be between 4 and 5
# * Now will check Silhouette method 

# In[32]:


sil_score = []
cluster_list = list(range(2, 10))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((subset_scaled_df))
    # centers = clusterer.cluster_centers_
    score = silhouette_score(subset_scaled_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.title("Selecting k with the Silhouette Score", fontsize=20)


# * From the silhouette scores, it looks like 4 would be the better value for K.

# In[33]:


# finding optimal number of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(4, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()


# In[34]:


# finding optimal number of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(5, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()


# In[35]:


# finding optimal number of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(6, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()


# In[36]:


# 4 seems to be the appropriate number of clusters
kmeans = KMeans(n_clusters=4, random_state=1)
kmeans.fit(subset_scaled_df)


# In[37]:


# adding kmeans cluster labels to the original dataframe
df["K_means_segments"] = kmeans.labels_


# In[38]:


cluster_profile = df.groupby("K_means_segments").mean()


# In[39]:


cluster_profile["count_in_each_segment"] = (
    df.groupby("K_means_segments")["Avg_Credit_Limit"].count().values
)


# In[40]:


# boxplots to visualize numerical variables for each cluster
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(5):
    sns.boxplot(ax=axes[ii], y=df[num_cols[counter]], x=df["K_means_segments"])
    counter = counter + 1

fig.tight_layout(pad=2.0)


# # Hierarchical Clustering
# * We already have the scaled subset dataset so we will perform algorithms using various distance and linkage metrics
# * These methods are used because we have lower dimensional data

# In[41]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(subset_scaled_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(subset_scaled_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm


# In[42]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# * Citybock and average linkage has the second highest cophenetic correlation at 0.8963.  We will look at different methods using Eucledian.

# In[43]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(subset_scaled_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(subset_scaled_df))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm


# * The highest cophenetic correlation is still Eucledian with average linkage.  
# * We will look at dendogram of the different linkage methods to see how everything is divided
# 

# In[44]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(subset_scaled_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(subset_scaled_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# * Looking at the dendogram, Ward gave good separate and distinct clusters but he cophenetic correlation was much lower than the rest. 
# * We will go with average linkage, 4 looks like the appropriate number of clusters

# In[63]:


# Creating 4 hierarchical clusters using the model with the best metrics and dendogram
HCmodel = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average")
HCmodel.fit(subset_scaled_df)


# In[64]:


# adding hierarchical cluster labels to the original and scaled dataframes
subset_scaled_df["HC_Cluster"] = HCmodel.labels_
data["HC_Clusters"] = HCmodel.labels_


# In[65]:


cluster_profile2 = data.groupby("HC_Clusters").mean()


# In[66]:


cluster_profile2["count_in_each_segments"] = (
    data.groupby("HC_Clusters")["Avg_Credit_Limit"].count().values
)


# In[49]:


# Copying data for outlier treatment clustering
data2 = df.copy()


# In[50]:


data2.duplicated().sum()


# In[51]:


data2 = data2[(~data2.duplicated())].copy()


# In[52]:


data2.shape


# In[53]:


# Finding the outliers in the next two cells
quartiles = np.quantile(
    data2["Avg_Credit_Limit"][data2["Avg_Credit_Limit"].notnull()], [0.25, 0.75]
)
limit_4iqr = 4 * (quartiles[1] - quartiles[0])
print(f"Q1 = {quartiles[0]}, Q3 = {quartiles[1]}, 4*IQR = {limit_4iqr}")


# In[54]:


outlier_limit = data2.loc[
    np.abs(data2["Avg_Credit_Limit"] - data2["Avg_Credit_Limit"].mean()) > limit_4iqr,
    "Avg_Credit_Limit",
]

outlier_limit


# In[55]:


# Code to drop the outliers from the dataset
df.drop(outlier_limit.index, axis=0, inplace=True)


# In[56]:


# Scaling a sthe data set before clustering
scaler = StandardScaler()
subset2 = data[num_cols].copy()
subset_scaled2 = scaler.fit_transform(subset)


# In[57]:


subset_scaled_df2 = pd.DataFrame(subset_scaled2, columns=subset2.columns)


# In[58]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(subset_scaled_df2, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(subset_scaled_df2))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm


# * There doesn't appear to be any difference upon removing outliers
# * PCA isn't needed because there are only 5 dimensions, there is not a need to reduce

# ## Cluster Profiling

# In[59]:


# Displaying cluster profiles and highlighting highest values in each cluster
cluster_profile.style.highlight_max(color="lightgreen", axis=0)


# In[60]:


# Code to plot numerical variables in each cluster
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Barplot of numerical variables for each cluster")
counter = 0
for ii in range(5):
    sns.barplot(
        ax=axes[ii],
        y=cluster_profile[cluster_profile.columns[counter]],
        x=cluster_profile.reset_index()["K_means_segments"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[67]:


# let's display cluster profiles
cluster_profile2.style.highlight_max(color="lightgreen", axis=0)


# In[71]:


# Code to plot numerical variables in each cluster
fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Barplot of numerical variables for each cluster")
counter = 0
for ii in range(5):
    sns.barplot(
        ax=axes[ii],
        y=cluster_profile2[cluster_profile.columns[counter]],
        x=cluster_profile2.reset_index()["HC_Clusters"],
    )
    counter = counter + 1

fig.tight_layout(pad=2.0)


# # Cluster Comparison and insights
#  
# ## K-means: 4 clusters 
#  
# ## Hiearchical cluster: euclidean distance metrics and average linkage</b>

# <b> The two different sets of cluster models were pretty similar.  
# 
# Both K-means and H-clusters, cluster 0 had very similar values for all clusters.</b>
# * Around 220 obervations
# * Average credit limit at $12K
# * Highest number of calls made
# * Total credit cards between 2-3
# 
# Customers having a lower credit limit and less cards make the most calls, and the assumption is that it is to customer service.  This group does use online banking moderately and doesn't visit the bank much.
# 
# 
# <b>K-means cluster 2 and HC cluster 1 were almost identical.</b>
# * 50 observations
# * Highest credit limit 
# * HIghest amount of credit cards
# * highest number of visits online
# * lowest number of visits to the bank
# * lowest number of calls to bank
# 
# This is the group that is completely comfortable with online banking and are mostly satisfied as calls and visits are very low.  They have the highest credit limit and number of cards.  They are tech savy and most likely higher income earners.
# 
# <b>K-Means model cluster 1, 3 and HC model cluster 2 have some similar data.</b>
# * Average credit card limit between 31K - 35K
# * Total credit cards, 5
# * K-means cluster 3 showed the highest bank visits
# 
# This cluster is somwhere in between.  Not tech savy as they rarely bank online,  moderate to high credit limit and moderate number of credit cards.  They don't contact the bank too often by phone but do so more in person.
# 

# # Recommendations
# <b>There needs to be more research/data in order to extract customer satisfaction insight.</b>  
#   * It is assumed that calls are made to customer service to complain or ask questions.
#   * It is also assumed that visits could partially be the same or to make deposits, withdrawals and general banking needs
# 
# <b>The Average credit amount used for each customer needs to be part of this research to compare the limit to the spending.</b>
#   - It is assumed that the credit limit given is tied to credit score and income earned so increasing limits may not always be available for every customer... secured cards
#   - Having amount used against amount available allows us to see the ratio and separate those customers that only use a small portion of their limits. This way we can better target those customers.
#   
# <b>Customer income is another very important variable not included in this dataset that should be.</b>
#   - Having this information will help us better target new customers whose income profile fits that of K-means cluster 2. The tech savvy, high credit limits, low calls and visits to bank type of customer. Self sufficient. 
#   - Important to note, it is unknown if this clusters online visits are purely banking, chatting with customer service, or some combination.  This is valuable information that should be added to the next analysis.

# ## Conclusion
# <b>Based on the data clusters and assumptions mentioned above:</b>
# * Follow up via phone call to customers with a credit limit around 12K and who made more than 4 calls to the bank to check in and see how they can improve their customer service.
# * Try to target new customers with higher income levels and very good credit scores via email, offering special low interest cards, possibly with rewards.  Since they are low risk customers.
# * Send personalized VIP emails to current AllLife customers that have credit limits over 140K and possess 8 credit cards, thanking them for their being a valued customer and offereing them incentives on new purchases. Also send personalized customer satisfaction surveys to this group to see what love about the bank and how they could do better.  This group is smaller so you can even offer to set up a call or live meeting.
# * Set up a credit card customer service specialist in the bank and randomly choose customers to complete a satisfaction survey.

# In[ ]:




