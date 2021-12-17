from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# Un transformador para remover columnas indeseadas
class DropDuplicados(BaseEstimator, TransformerMixin):           

    def fit(self, X):
        return self

    def transform(self, X):
        data = X.copy()        
        return data.drop_duplicates()


class ConvertirColumnasCategoricas(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def convertirCategoricas(self,df_base):
      for columna in self.columns:
        df_base[columna] = df_base[columna].astype('category')
      return df_base

    def transform(self, X):
        # Primero realizamos la cópia del DataFrame 'X' de entrada
        data = X.copy()        
        return self.convertirCategoricas(data)

class DataScaleImputer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()        
        transformer = ColumnTransformer(transformers=[('scaler', StandardScaler(),self.columns)])
        X_transform = transformer.fit_transform(data)
        X_imputed_df = pd.DataFrame(data = X_transform, columns = self.columns)
        data[self.columns] = X_imputed_df[self.columns]
        return data

class DataOneHotEncoderTransform(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X):
        return self
    
    def transform(self, X):    
        data = X.copy()              
        return pd.get_dummies(data, columns=self.columns, drop_first=False)


def convert_binary(x):
    if x == 'T':
        return 1
    else:
        return 0

class ConvertBinaryTransform(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X):
        return self
    
    def transform(self, X):    
        data = X.copy()              
        data[self.columns] = data[self.columns].applymap(convert_binary)    
        return data


class DataLabelEncoderTransform(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X):
        return self
    
    def transform(self, X):    
        df_temp = X.copy()      
        encoder = LabelEncoder()
        df_encoder= df_temp[self.columns].apply(encoder.fit_transform)             
        X[self.columns] = df_encoder
        return X

