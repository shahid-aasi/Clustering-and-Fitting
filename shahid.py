import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def file_reader(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    df = pd.read_csv(fn, skiprows=4)
    df = df.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return df


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1990))
    return f


def Data_for_Country(df, country_name, start_year, end_year):
    """
    To Get Data of Specific Countries
    """
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    df.index = df.index.astype(int)
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def plot_silhouette_score(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(
            2,
            max_clusters +
            1),
        silhouette_scores,
        marker='s',
        color='r')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


Arable_land_of_land_area = file_reader('Arable_land_%_of_land_area.csv')
Forest_area_of_land_area = file_reader('Forest_area_%_of_land_area.csv')
country = 'India'
df_Ar = Data_for_Country(Arable_land_of_land_area, 'India', 1990, 2020)
df_FA = Data_for_Country(Forest_area_of_land_area, 'India', 1990, 2020)

# df_fit, df_min, df_max = ct.scaler(df_cluster)

df = pd.merge(df_Ar, df_FA, left_index=True, right_index=True)
df = df.rename(
    columns={
        country +
        "_x": 'Arabel Land',
        country +
        "_y": 'Forest_Area'})
df_fit, df_min, df_max = ct.scaler(df)
plot_silhouette_score(df_fit, 12)

nc = 2  # number of cluster centres
kmeans = KMeans(n_clusters=nc, n_init=10, random_state=0)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure()
# scatter plot with colours selected using the cluster numbers
# now using the original dataframe
scatter = plt.scatter(
    df['Arabel Land'],
    df['Forest_Area'],
    c=labels,
    cmap="Dark2")
# colour map Accent selected to increase contrast between colours
# rescale and show cluster centres
scen = ct.backscale(cen, df_min, df_max)
xc = scen[:, 0]
yc = scen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.xlabel('Arabel Land')
plt.ylabel('Forest Area')
plt.title('Arable Land vs Forest Area In India ')
plt.savefig('Clustering_plot.png', dpi=300)
plt.show()

popt, pcorr = opt.curve_fit(exp_growth, df_Ar.index, df_Ar[country],
                            p0=[4e3, 0.001])
# much better
df_Ar["pop_exp"] = exp_growth(df_Ar.index, *popt)
plt.figure()
plt.plot(df_Ar.index, df_Ar[country], label="data")
plt.plot(df_Ar.index, df_Ar["pop_exp"], label="fit", color='plum')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Arabel Land')
plt.title('Arabel Land in India 1990-2020')
plt.savefig(country + '.png', dpi=300)
years = np.linspace(1995, 2030)
pop_exp = exp_growth(years, *popt)
sigma = err.error_prop(years, exp_growth, popt, pcorr)
low = pop_exp - sigma
up = pop_exp + sigma
plt.figure()
plt.plot(df_Ar.index, df_Ar[country], label="data")
plt.plot(years, pop_exp, label="Forecast", color='plum')
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color="plum")
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Arabel Land')
plt.title('Arabel Land in India Forecast')
plt.savefig('forecast.png', dpi=300)
plt.show()

popt, pcorr = opt.curve_fit(exp_growth, df_FA.index, df_FA[country],
                            p0=[4e3, 0.001])
# much better
print("Fit parameter", popt)
df_FA["pop_exp"] = exp_growth(df_Ar.index, *popt)
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data")
plt.plot(df_FA.index, df_FA["pop_exp"], label="fit", color='plum')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in India 1990-2020')
plt.savefig('data.png', dpi=300)
years = np.linspace(1995, 2030)
print(*popt)
pop_exp = exp_growth(years, *popt)
sigma = err.error_prop(years, exp_growth, popt, pcorr)
low = pop_exp - sigma
up = pop_exp + sigma
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data")
plt.plot(years, pop_exp, label="Forecast", color='plum')
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color="plum")
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in India Forecast')
plt.savefig('forecast.png', dpi=300)
plt.show()
