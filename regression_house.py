import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#import seaborn as sns
#import matplotlib.pyplot as plt


def fetch_df(data_csv):
    return pd.read_csv(data_csv)

def normalise_df(df):
    df_mean = df.mean()
    df_std = df.std()    
    df_norm = (df - df_mean) / (df_std)
    return df_norm, df_mean, df_std

def fetch_housing_dataset(data_csv='./housing.csv'):
    data = fetch_df(data_csv)
    # 11 columns total, 11 holds word mapping for 10
    X = data.iloc[:,0:-2]
    y = data.iloc[:,-2]
    labels = data.iloc[:,-1]
    return X, y, labels

def shuffle_split(X, y, test_split=0.2):
    X, y = shuffle(X, y)
    return train_test_split(X, y, test_size=test_split)

def plot_data(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plot = sns.PairGrid(df)
    plot = plot.map(plt.scatter)
    plt.show()

def plot_cost(J_history, num_iters):
    import matplotlib.pyplot as plt
    plt.plot([i for i in range(num_iters)], J_history)
    plt.show()

def hypothesis(X, theta):
    # linear: h(x) = theta0 + theta1*x
    # linear (matrix): h(x) = x*theta
    # quad:   h(x) = theta0 + theta1*x + theta2*(x**(2))
    # cubic:  h(x) = theta0 + theta1*x + theta2*(x**(2)) + theta3*(x**(3))
    h = X.dot(theta)
    return h

def calc_cost(X, y, theta):
    # mse: J(theta) = (1/2*m)*(sum(h(x)-y)**2)
    m = len(y)
    h = hypothesis(X, theta)
    J = (1/(2*m))*np.sum(np.square(h-y))
    return J

def run_bgd(X, y, theta, alpha, num_iters):
    J_history = []
    m = len(y)
    X = X
    y = y
    for i in range(num_iters):
        h = hypothesis(X, theta)
        delta = (1 / m) * np.sum(X.T.dot(h - y))
        theta -= alpha * delta
        cost = calc_cost(X, y, theta)
        print('iteration {} cost: {}'.format(i, cost))
        J_history.append(cost)
    return theta, J_history

if __name__ == '__main__':
    
    # fetch normalized data
    X, y, labels = fetch_housing_dataset()
    
    df = X
    df.insert(loc=X.shape[1], column='ocean_proximity_group', value=y)
    
    print(df)
    
    lat = df['latitude']
    long = df['longitude']
    prices = df['housing_median_age']
    
    #import seaborn as sns
    from matplotlib import pyplot as plt
    plt.scatter(long, lat, c=prices)
    plt.show()
    
    # shuffle the data and split off 20% for testing data
    #X, X_test, y, y_test = shuffle_split(X, y, 0.2)
    