# DeeplearningProject

DeeplearningProject is a kaggle competition.

# Steps

  ## Get the Data
    Weather Dataset
    Building Metadata
    Train Dataset
  ## Data visualisation
  ## Data preparation
    Fill Nan value in weather dataframe
    Categorize primary_use column in building Metadata dataframe ...
  ## Merge Datasets 
```python
data = df_train.merge(building, on=['building_id'], how='left')
data = data.merge(weather_train_df, on=['timestamp', 'site_id'], how='left')
```
    Add some feature.    
  ## Feature selection
  ## Split the data into train set and validation set 
  ```python
    from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data.meter_reading, test_size=0.3)
```
  ## build LSTM Architecture using keras  
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding
from keras.optimizers import RMSprop,Adam
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")

### Architecture LSTM

model = Sequential()

model.add(LSTM(100, activation = 'relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))

model.summary()
```
  ## Load the model
  ## Test

