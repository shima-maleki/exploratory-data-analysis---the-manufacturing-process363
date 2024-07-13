import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns


#df = pd.read_csv('data\data.csv')
#print(df.head())

#print()
#print(df.columns)
#task 1
class DataTransform:
    def __init__(self,file_path):
        self.file_path = file_path

    def read_data(self):
        df = pd.read_csv(self.file_path)
        return df
    
    def transform_columns(self):
        df = self.read_data()
        columns = list(df.columns)
        new_columns = []
        for column in columns:
            col = column.split()
            new_column = '_'.join(col)
            new_columns.append(new_column)

        df.columns = new_columns
        return df



dt = DataTransform(file_path = 'data\data.csv')   
df =  dt.transform_columns()

#task2
class DataFrameInfo:
    def __init__(self,data ):
        self.data = data

    def columns_dtype(self):
        info = self.data.info()
        print(info)
        return list(info)

    def stats_values(self):
        mean_value =df.mean()
        median_value = df.median()
        std_value = df.std()
        print(f'mean value: {mean_value}')
        print(f'median value : {median_value}')
        print(f'standatd deviation: {std_value}')
        return mean_value, median_value,std_value
        
    def dist_values(self):
        unique_values = dict()
        for col in self.data.columns:
            if self.data[col].dtype == 'o':
                unique_value = self.data.col.nunique()

                unique_values[col] = unique_value
        print(unique_values)
        return unique_values  

    def print_shape(self):
        print(self.data.shape)       


    def cal_null_percentage(self):
        null_percentage = self.data.isnull().mean() * 100
        print(null_percentage)
        return null_percentage

#task3 
class Plotter(DataFrameInfo):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        
    def plot_missing_values(self):
        null_percentage = self.cal_null_percentage()
        plt.bar(self.data.columns, null_percentage)
        plt.xticks(rotation=90)
        plt.ylabel('Percetage of missing values')
        plt.title('Missing Values')
        plt.savefig('images\missing_value.png')
        plt.show()


    def plot_stats(self):
        mean_value, median_value, std_value = self.stats_values()
        plt.bar(mean_value.index, mean_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('mean values')
        plt.title('Mean Values')
        plt.savefig('images\mean_value.png')  
        plt.show()  

        plt.bar(median_value.index, median_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('median values')
        plt.title('Median Values')
        plt.savefig('images\median_value.png')
        plt.show()

        plt.bar(std_value.index, std_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('std values')
        plt.title('STD Deviation')
        plt.savefig('images\std_value.png')
        plt.show()

    def plot_skewness(self):
        skewness = self.data.skew()
        plt.bar(skewness.index, skewness.values)
        plt.xticks(rotation=90)
        plt.ylabel('skewness values')
        plt.title('Skewness')
        plt.savefig('images\skewness_value.png')
        plt.show()

    def skewness_histogram(self, column):
        histogram = sns.histplot(self.data[column], label="Skewness: %.2f"%(self.data[column].skew()) )
        histogram.legend()
        return histogram
    
    def correlation_matrix(self):
        corr = self.data.corr()
        print(corr)

        cmap = sns.color_palette("viridis", as_cmap=True) 

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, square=True, linewidths=.5, annot=True, cmap=cmap, fmt=".2f")
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.savefig('images\correlation.png')
        plt.show()


pt = Plotter(df)
pt.plot_missing_values() 
pt.plot_stats()
pt.plot_skewness()
pt.correlation_matrix()

class DataFrameTransform:
    def __init__(self, data):
        self.data = data

    def handle_null_values(self):
        null_percentage = self.data.isnull().mean() * 100
        for col, value in zip(null_percentage.index, null_percentage.values):
            if value > 0:
                if self.data[col].dtype == 'O':
                    self.data[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        return self.data
    
    def box_cox_transform(self, column):

        boxcox_column = stats.boxcox(self.data[column])
        boxcox_column = pd.Series(boxcox_column[0])
        return boxcox_column

    def yeo_johnson_transform(self, column):
        yjh_column = stats.yeojohnson(self.data[column])
        yjh_column = pd.Series(yjh_column[0])
        return yjh_column

    def remove_outlier(self, column, z_score_threshold):
        mean = np.mean(self.data[column])
        std = np.std(self.data[column]) 
        z_scores = (self.data[column] - mean) / std
        abs_z_scores = pd.Series(abs(z_scores)) 
        mask = abs_z_scores < z_score_threshold
        DataFrame = self.data[mask]       
        return DataFrame
    
dft = DataFrameTransform(df)
dft.handle_null_values()








        




 
        










                