from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import Union, Any
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class TransformedPredictClassifier(BaseEstimator, TransformerMixin):
    """A meta estimator that both transforms the response variable.
    
    Args:
        classifier: An instance of a class that inherits from BaseEstimator
        transformer: An instance of a class that inherits from TransformerMixin

    """
    def __init__(self, columnas_clusters_kmeans,columnas_productos):
        self.columnas_clusters_kmeans = columnas_clusters_kmeans
        self.columnas_productos = columnas_productos
        self.col_reglas_orden = ['confidence','lift']
    
    def ordernar_lista(self, lista):
        lista_temp = [v for v in lista if v in self.columnas_productos]  
        return sorted(lista_temp,key=self.columnas_productos.index)

    def validar_antecedente(self, row,fvalor):
        valor_antecente = row['antecedents']  
        resultado = valor_antecente.issubset(fvalor)&valor_antecente.issuperset(fvalor)
        return resultado

    def crear_recomendaciones(self,row,reglas):
        RECOMMENDATION_1=''
        CONFIDENCE_1=''
        RECOMMENDATION_2=''
        CONFIDENCE_2=''
        RECOMMENDATION_3=''
        CONFIDENCE_3=''
        LIFT_1=-1
        SUPPORT_1=-1
        LIFT_2=-1
        SUPPORT_2=-1
        LIFT_3=-1
        SUPPORT_3=-1
        NRO_REGLAS = len(reglas)  
        fvalor_prod = frozenset(row['set_productos'])  
        df_reglas_prod=reglas[reglas.apply(lambda x:self.validar_antecedente(x,fvalor_prod), axis = 1)]
        fvalor_dat = frozenset(row['set_datos'])  
        df_reglas_dat=reglas[reglas.apply(lambda x:self.validar_antecedente(x,fvalor_dat), axis = 1)]
        df_reglas_temp = pd.concat([df_reglas_prod,df_reglas_dat]).drop_duplicates()
        df_reglas = df_reglas_temp.sort_values(self.col_reglas_orden, ascending =[False, False]).head(3)
        #print(df_reglas)
        if len(df_reglas)>0:
            nro_reglas = len(df_reglas)
            for i in range(0,nro_reglas):
                #print(i)
                if i==0:
                    lista_ordenada = self.ordernar_lista(list(df_reglas.iloc[i]['consequents']))        
                    RECOMMENDATION_1 = ",".join(lista_ordenada)
                    CONFIDENCE_1 = df_reglas.iloc[i]['confidence']
                    LIFT_1 = float(df_reglas.iloc[i]['lift'])
                    SUPPORT_1 = float(df_reglas.iloc[i]['support'])
                elif i==1:
                    lista_ordenada = self.ordernar_lista(list(df_reglas.iloc[i]['consequents']))
                    RECOMMENDATION_2 = ",".join(lista_ordenada)
                    CONFIDENCE_2 = df_reglas.iloc[i]['confidence']
                    LIFT_2 = float(df_reglas.iloc[i]['lift'])
                    SUPPORT_2 = float(df_reglas.iloc[i]['support'])
                elif i==2:
                    lista_ordenada = self.ordernar_lista(list(df_reglas.iloc[i]['consequents']))
                    RECOMMENDATION_3 =  ",".join(lista_ordenada)
                    CONFIDENCE_3 = df_reglas.iloc[i]['confidence']
                    LIFT_3 = float(df_reglas.iloc[i]['lift'])
                    SUPPORT_3 = float(df_reglas.iloc[i]['support'])
                else:
                    break          
        return RECOMMENDATION_1,CONFIDENCE_1,LIFT_1,SUPPORT_1,RECOMMENDATION_2,CONFIDENCE_2,LIFT_2,SUPPORT_2,RECOMMENDATION_3,CONFIDENCE_3,LIFT_3,SUPPORT_3,NRO_REGLAS

    def crear_reglas_cluster_id(self,df_clusters_merge, nombre_cluster):
        tipos_clusters = df_clusters_merge[nombre_cluster].value_counts()
        print(tipos_clusters)
        lista_rules = []
        for id_cluster in tipos_clusters.index:       
            df_temp_cluster = df_clusters_merge[df_clusters_merge[nombre_cluster]==id_cluster]
            print("Cluster: {fclus} - Tamanio: {fsize}".format(fclus=id_cluster,fsize=len(df_temp_cluster)))      
            print(len(df_temp_cluster[self.columnas_productos].columns))
            frequent_itemsets_temp = apriori(df_temp_cluster[self.columnas_productos], min_support=0.10, use_colnames=True,max_len=5)
            rules_mlxtend_temp = association_rules(frequent_itemsets_temp,metric="confidence", min_threshold=0.8)
            restulados_sort_temp = rules_mlxtend_temp.sort_values(self.col_reglas_orden, ascending =[False, False])  
            print("Tamanio items: {fitem} - Tamanio Reglas Ordenadas: {freg}".format(fitem=len(frequent_itemsets_temp),freg=len(restulados_sort_temp )))
            lista_rules.append(restulados_sort_temp)
        return lista_rules    
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()        
        return data

    def predict(self,X):     
        data = X.copy()                        
        lista_reglas = []
        columnas_recomendacion = ['RECOMMENDATION_1','CONFIDENCE_1','LIFT_1','SUPPORT_1',
         'RECOMMENDATION_2','CONFIDENCE_2','LIFT_2','SUPPORT_2',
         'RECOMMENDATION_3','CONFIDENCE_3','LIFT_3','SUPPORT_3','NRO_REGLAS']
        for columna_mod_kmeans in self.columnas_clusters_kmeans:
            lista_reglas_temp = self.crear_reglas_cluster_id(data,columna_mod_kmeans)
            lista_reglas.extend(lista_reglas_temp)
        df_rules = pd.concat(lista_reglas).sort_values(self.col_reglas_orden, ascending =[False, False])
        df_rules = df_rules.drop_duplicates(['antecedents','consequents'], keep='first')   
        df_respuetas_final = data.copy()
        df_respuetas_final[columnas_recomendacion] = df_respuetas_final.apply(lambda row : self.crear_recomendaciones(row,df_rules), axis = 1, result_type ='expand')
        return df_respuetas_final
  
