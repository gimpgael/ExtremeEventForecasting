"""
Algorithm to forecast time series. Based on a time series, the algorithm:
    - Generates a set of features, pre-coded
    - Compresses these data through an autoencoder
    - Inputs then the data into a LSTM model
Note that the code is quite raw, so do not hesitate to make the modifications
directly in the class as you wish.

The idea of this approach comes from a Uber article, linked below, 
https://eng.uber.com/neural-networks/

Also linked to a presentation
https://forecasters.org/wp-content/uploads/gravity_forms/7-c6dd08fee7f0065037affb5b74fec20a/2017/07/Laptev_Nikolay_ISF2017.pdf

itself linked to another paper:
https://robjhyndman.com/papers/icdm2015.pdf

The idea was to develop time series approach in order to model extrem events. I
thought that this kind of approach was interesting in the case of financial
time series, so decided to implement it. The difference in between the papers
and the below implementation concern the seasonality. Indeed the frequency of 
data is not the same, we are designing here to work in a daily basis, so the
seasonality and its derivatives are not relevant as features, and so are not
coded.

"""

# Import librairies
import numpy as np
import pandas as pd

import statsmodels.api as sm

from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model, Sequential, load_model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

np.random.seed(42)


class TimeSeries_Ub():
    """Time Series modelling based on the initial Uber article. The first part
    is about the features building, and then the modelling itself. Note that 
    the method to create models are almost externals at the class, so they are
    called here but the user needs to manually input the parameters (the number
    of hidden layers, length of timesteps ...)
    
    Attributes
    -----------------   
    - window: rolling window of the time series used to create variables
    - autoencoder_activation: activation function to be used by the autoencoder
    - model_activation: activation function to be used by the model
    
    Usage
    - initialize_dataset: computes the time serie features
    - sequential_autoencoder: autoencoder
    - create_variable_for_model: create the variable to be inputed in the 
    recurrent neural network
    - forecast_model: create LSTM model
        
    """
    
    def __init__(self, window, autoencoder_activation, model_activation):
        """Initialize the algorithm, with a fixed rolling window
        :param window: (int)
        """
        self.window = window
        self.autoencoder_activation = autoencoder_activation
        self.model_activation = model_activation
        
    def _extract_time_series(self, x, i, taille):
        """Function extracting the sub time series from a pandas Series. We use
        this function  because the pandas Series does not have same properties
        as numpy array.
        :param x: (pandas Series) time series
        :param i: (int) index 
        :param taille: (int)
        """
        
        # Need to differentiate the last case
        if i != taille:
            x_int = x[x.index[i-self.window]:x.index[i]]
        else:
            x_int = x[x.index[i-self.window]:]
        
        return x_int
        
    def ts_mean(self, x):
        """Computes the time series mean for a rolling window
        :param x: (pandas Series) time series
        """
        x.name = 'average'
        return x.rolling(self.window).mean()
    
    def ts_var(self, x):
        """Computes the time series variance for a rolling window
        :param x: (pandas Series) time series
        """
        x.name = 'variance'
        return x.rolling(self.window).var()
    
    def ts_autocorrelation(self, x):
        """Computes the time series autocorrelation for a rolling window
        :param x: (pandas Series) time series
        """
        
        # Initialize output
        res = pd.Series(data = None, index = x.index, name = 'autocorr')
        
        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            res[res.index[i-1]] = self._extract_time_series(x, i, x.shape[0]).autocorr()
                        
        return res
    
    def ts_entropy(self, x):
        """Computes the time series entropy for a rolling window
        :param x: (pandas Series) time series
        """
        
        # Initialize output
        res = pd.Series(data = None, index = x.index, name = 'entropy')
        
        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            res[res.index[i-1]] = entropy(self._extract_time_series(x, i, x.shape[0]))
                
        return res
    
    def ts_trend(self, x):
        """Computes the trend coefficient and the variance of the residuals of
        the time series versus the trend, for a rolling window
        :param x: (pandas Series) time series
        """
        
        # Initialize outputs
        res1 = pd.Series(data = None, index = x.index, name = 'trend')
        res2 = pd.Series(data = None, index = x.index, name = 'trend_p_value')
        res3 = pd.Series(data = None, index = x.index, name = 'residuals_variance')
        
        # Create the X, to compute the trend
        x_int = np.arange(self.window)
        x_int = sm.add_constant(x_int)

        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            
            # Extract the corresponding time series window
            y_int = self._extract_time_series(x, i, x.shape[0])
                        
            # Linear Model
            m = sm.OLS(y_int, x_int)
            results = m.fit()
            
            # Trend value and its p-value
            res1[res1.index[i-1]] = results.params['x1']
            res2[res2.index[i-1]] = results.pvalues['x1']
            
            # Strength of trend
            res3[res3.index[i-1]] = np.var(results.predict(x_int) - y_int)
            
        return res1, res2, res3
    
    def ts_spike(self, x):
        """Computes the spike of a time series, for a rolling window
        :param x: (pandas Series) time series
        """
        
        # Initialize output
        res = pd.Series(data = None, index = x.index, name = 'spike')
        
        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            
            # Extract the corresponding time series window and computes the spike
            x_int = self._extract_time_series(x, i, x.shape[0])
            res[res.index[i-1]] = self.function_spike(x_int)
            
        return res
    
    def function_spike(self, x):
        """Computes the spike function for a time series
        :param x: (pandas Series) time series
        """
        
        # Computes the max and min spikes versus the average
        M, m = x.max() - x.mean(), x.min() - x.mean()
        
        # Return the biggest spike    
        return M if np.abs(M) > np.abs(m) else m
    
    def ts_crossing_points(self, x):
        """Computes the number of times the mean is crossed by the time series 
        for a rolling window
        :param x: (pandas Series) time series
        """
        
        # Initialize output
        res = pd.Series(data = None, index = x.index, name = 'crossing_points')
        
        # Loop through time series
        for i in range(self.window, x.shape[0] + 1):
            
            # Extract the corresponding time series window and computes the crossing points
            x_int = self._extract_time_series(x, i, x.shape[0])
            res[res.index[i-1]] = self.function_crossing_points(x_int)
            
        return res
            
    def function_crossing_points(self, x):
        """Computes the number of times the times series cross its mean
        :param x: (pandas Series) time series
        """
        
        # Reduce the mean
        x = np.sign(x - x.mean())
            
        # Number of times the mean is crossed
        x = x.diff().fillna(0)
        return np.abs(x).sum() / 2

    def compute_time_series_features(self, x):
        """Computes the different features of the time series
        :param x: (pandas Series) time series
        """
        
        # In case, transform the initial input into a pandas Series
        x = pd.Series(x, name = 'time_series')
        
        # Computes the individual features
        x1 = self.ts_mean(x)
        x2 = self.ts_var(x)
        x3 = self.ts_autocorrelation(x)
        x4 = self.ts_entropy(x)
        x5, x6, x7 = self.ts_trend(x)
        x8 = self.ts_spike(x)
        x9 = self.ts_crossing_points(x)
        
        # Concatenate them into a DataFrame
        res = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9], axis = 1)
        
        # Look also at the features changes
        res_diff = res.diff()
        res_diff.columns = [x + '_diff' for x in res_diff.columns]
        
        return pd.concat([res, res_diff], axis = 1)
    
    def initialize_dataset(self, y):
        """Build the dataset based on a time series feature y
        :param x: (pandas Series) time series
        """
        
        # Compute all variables
        X = self.compute_time_series_features(y)
        X = X.iloc[self.window-1:,:]
        y = y.iloc[self.window-1:]
        X = X.fillna(method='ffill').fillna(X.mean())
        
        return X, y
        
    def sequential_autoencoder(self, x, num_layers, timesteps = 5):
        """LSTM Autoencoder to extract high level features
        Note that the x input is still in a raw format [m,n] where m are dates 
        and n are the features. The sequence length is still given by timesteps,
        and some data manipulation needs to be made before inputing the data in
        the model. This is why there is the function create_variable_for_model
        :param x: (pandas Series) time series
        :param num_layers: (int) number of layers
        :param timesteps: (int) depth of time series for the LSTM cell
        """
        
        # Input
        inputs = Input(shape = (timesteps, x.shape[-1]))
        
        # Encoder
        encoder = LSTM(num_layers, activation = self.autoencoder_activation)(inputs)
        
        # Decoder
        decoder = RepeatVector(timesteps)(encoder)
        decoder = LSTM(x.shape[-1], return_sequences = True, activation = self.autoencoder_activation)(decoder)
        
        # Models
        sequential_autoencoder = Model(inputs, decoder)
        sequential_autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
        encoder = Model(inputs, encoder)
        encoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        return sequential_autoencoder, encoder
    
    def prepare_data_autoencoder(self, X, y):
        """Prepare the data overall, by scaling
        :param X: (pandas DataFrame) features
        :param y: (pandas Series) time series
        """
        
        # Initialize the two scalers
        sc_x, sc_y = StandardScaler(), StandardScaler()
        
        # Convert data into proper matrix for scaling, and transform them
        X_values, y_values = X.values, y.values.reshape(-1,1)
        X_values, y_values = sc_x.fit_transform(X_values), sc_y.fit_transform(y_values)
        
        return X_values, y_values, sc_x, sc_y
        
    def generate_encoder(self, X, num_layers, timestep, epochs=200):
        """Generate and fit the autoencoder. It returns the model with the best
        validation metrics. The compression dimension is num_layers.
        :param X: (pandas DataFrame)
        :param: num_layers: (int)
        :param timestep: (int)
        :param: epochs: (int)
        """
        
        checkpoint = ModelCheckpoint('autoencoder.h5', 
                                     monitor = 'val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min')
        autoencoder, encoder = self.sequential_autoencoder(X, num_layers, timestep)
        autoencoder.fit(X, X, batch_size=256, 
                              epochs=epochs, 
                              validation_split=0.2, 
                              callbacks=[checkpoint])
        
        autoencoder = load_model('autoencoder.h5')
        
        # Return only the encoder part
        return Model(autoencoder.layers[0].input, autoencoder.layers[1].output)
        
    def forecast_model(self, x, num_layers, timesteps):
        """LSTM model for forecasting. 
        Note that the x input is still in a raw format [m,n] where m are dates 
        and n are the features. The sequence length is still given by timesteps,
        and some data manipulation needs to be made before inputing the data in
        the model. This is why there is the function create_variable_for_model
        :param x: (pandas DataFrame) features
        :param num_layers: (int) number of layers
        :param timesteps: (int) depth of time series for the LSTM cell
        """
        
        # LSTM Model
        model = Sequential()
        model.add(LSTM(units = num_layers, input_shape = (timesteps, x.shape[-1]),
                       activation = self.model_activation))
        model.add(Dense(1, activation = None))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        return model
    
    def create_variable_for_model(self, x, timesteps):
        """Transform the variables inputs into an appropriate format for the 
        second model. The x input needs to be already in a numpy format.
        :param x: (numpy array) features
        :param timesteps: (int) depth of time series for the LSTM cell        
        """
        
        # Initialize output
        res = []
        
        # Create the output
        for idx in range(x.shape[0] - timesteps):
            res.append(x[idx:idx+timesteps,:])
        
        return np.array(res)
    
    
if __name__ == '__main__':

    # -----------------------
    # Dashboard
    time_window = 5
    autoencoder_activation = 'tanh'
    model_activation = 'tanh'
    n_layers_autoencoder = 4
    n_units = 5
    # -----------------------
    
    # Generate a random time series
    y = pd.Series(np.random.randint(low=1, high=100, size=1000))
        
    alg = TimeSeries_Ub(time_window, autoencoder_activation, model_activation)
    X, y = alg.initialize_dataset(y)
    
    # Scale down the data
    x_std, y_std, sc_x, sc_y = alg.prepare_data_autoencoder(X, y)
    x_train = alg.create_variable_for_model(x_std, time_window)
    y_train = y_std[time_window:]

    # Dimensionality reduction
    encoder = alg.generate_encoder(x_train, n_layers_autoencoder, time_window)
    x_train = encoder.predict(x_train)
    
    x_train = alg.create_variable_for_model(x_train, time_window)
    y_train = y_train[time_window:]

    model = alg.forecast_model(x_train, n_units, time_window)
    history = model.fit(x_train, y_train)