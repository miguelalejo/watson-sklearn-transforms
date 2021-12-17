from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import Union, Any

class TransformedTargetClassifier(ClusterMixin, BaseEstimator, TransformerMixin):
    """A meta estimator that both transforms the response variable.
    
    Args:
        classifier: An instance of a class that inherits from BaseEstimator
        transformer: An instance of a class that inherits from TransformerMixin

    """
    def __init__(self, column, classifier: BaseEstimator, transformer: Union[TransformerMixin, Pipeline]):
        self.classifier = classifier        
        self.column = column
        self.transformer = transformer
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        data = X.copy() 
        df_data_base =  data[[self.column,'long_productos']]
        df_modelo_data = self.transformer.transform(df_data_base)
        modelo_saved = self.classifier
        df_modelo_data['Cluster'] = modelo_saved.predict(df_modelo_data)
        return pd.concat([data,df_modelo_data['Cluster']],axis=1)
  
