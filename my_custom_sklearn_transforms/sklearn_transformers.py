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

    def fit(self, X, y=None):
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

class CrearNuevasCategoriasTransform(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X):
        return self
    
  
    def transform(self, X):
      data = X.copy()
      data["RANGO_Idade_RANGO_Renda"] = (data['RANGO_Idade'].astype(str) + "_" +data['RANGO_Renda'].astype(str))           
      data['Regiao_RANGO_Idade'] = (data['Regiao'].astype(str)+ "_" +data['RANGO_Idade'].astype(str))
      data['Regiao_RANGO_Renda'] = (data['Regiao'].astype(str)+ "_" +data['RANGO_Renda'].astype(str))      
      return data

def crear_transacciones(row,columnas_productos,columnas_transacciones):
  df_temp_row = pd.DataFrame({'producto':row[columnas_productos].index,'estado':row[columnas_productos].values})
  valores = df_temp_row[(df_temp_row['estado']=='T') | (df_temp_row['estado']==True)| (df_temp_row['estado']==1) ]['producto'].values.tolist()
  df_data_row = pd.DataFrame({'datos':row[columnas_transacciones].index,'estado':row[columnas_transacciones].values})
  valores_data = df_data_row[(df_data_row['estado']=='T') | (df_data_row['estado']==True)| (df_data_row['estado']==1) ]['datos'].values.tolist()
  return valores, len(valores),valores_data,len(valores_data)


class ConvertCrearTransaccionesTransform(TransformerMixin):
    def __init__(self, columns, transacciones):
        self.columns = columns
        self.transacciones = transacciones
        
    def fit(self, X):
        return self
    
    def transform(self, X):    
        columnas_ref_prod = ['set_productos','long_productos','set_datos','long_datos']
        data = X.copy()              
        data[columnas_ref_prod] = data.apply(lambda row : crear_transacciones(row,self.columns,self.transacciones), axis = 1, result_type ='expand')        
        return data

class CrearNuevosRangosColumnas(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def crearRangoEdad(self, df_base):
      pd_serie_age_temp = df_base['Idade']
      conditions  = [  (pd_serie_age_temp>=15)&(pd_serie_age_temp<25), (pd_serie_age_temp>=25)&(pd_serie_age_temp<35),
                    (pd_serie_age_temp>=30)&(pd_serie_age_temp<45),(pd_serie_age_temp>=45)]
      choices     = [  '15-24', '25-35','35-45','>=45']         
      df_base['RANGO_Idade'] = np.select(conditions, choices, default="DESCONOCIDO")      
      return df_base     


    def crearRangoIngreso(self, df_base):
      pd_serie_temp = pd.to_numeric(df_base['Renda'], errors='coerce')
      conditions  = [ (pd_serie_temp<=7500),
              (pd_serie_temp>=7500)&(pd_serie_temp<8000),(pd_serie_temp>=8000)&(pd_serie_temp<8500),
              (pd_serie_temp>=8500)  ]
      choices     = ['<=7500', 
                     '7500-8000', '8000-8500',                     
                     '>=8500']    
      df_base['RANGO_Renda'] = np.select(conditions, choices, default="DESCONOCIDO")
      return df_base   
          
    def transform(self, X):
        # Primero realizamos la cópia del DataFrame 'X' de entrada
        data = X.copy()     
        dfRangoEdad = self.crearRangoEdad(data)
        dfRangoIngreso = self.crearRangoIngreso(dfRangoEdad)
        return dfRangoIngreso 
