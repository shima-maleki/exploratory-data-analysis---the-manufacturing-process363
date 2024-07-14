import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
from db_utils import save_to_csv
import pandas as pd

class DataTransform:
    """
    A class used to transform data by reading from a CSV file and renaming columns.

    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the DataTransform object.

        Parameters
        ----------
            file_path : str
                The file path to the CSV file.
        """
        self.file_path = file_path

    def read_data(self):
        """
        Reads the CSV file from the file path.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data from the CSV file.
        """
        df = pd.read_csv(self.file_path)
        return df
    
    def transform_columns(self):
        """
        Transforms the DataFrame's columns by replacing spaces with underscores.

        Returns
        -------
        pd.DataFrame
            A DataFrame with transformed column names.
        """
        df = self.read_data()
        columns = list(df.columns)
        new_columns = []
        for column in columns:
            col = column.split()
            new_column = '_'.join(col)
            new_columns.append(new_column)

        df.columns = new_columns
        return df


class DataFrameInfo:
    """
    A class used to provide various information and statistics about a DataFrame.

    Attributes
    ----------
    data : pd.DataFrame
        The DataFrame to analyze.
    """

    def __init__(self, data):
        """
        Constructs all the necessary attributes for the DataFrameInfo object.

        Parameters
        ----------
            data : pd.DataFrame
                The DataFrame to analyze.
        """
        self.data = data

    def columns_dtype(self):
        """
        Returns the data types of the DataFrame's columns.

        Returns
        -------
        pd.Series
            A Series containing the data types of the DataFrame's columns.
        """
        return self.data.dtypes

    def stats_values(self):
        """
        Returns the mean, median, and standard deviation of the DataFrame's columns.

        Returns
        -------
        tuple
            A tuple containing three Series: the mean, median, and standard deviation of the DataFrame's columns.
        """
        mean_value = self.data.mean()
        median_value = self.data.median()
        std_value = self.data.std()
        return mean_value, median_value, std_value

    def unique_values(self):
        """
        Returns the number of unique values in each column of the DataFrame.

        Returns
        -------
        dict
            A dictionary where the keys are column names and the values are the number of unique values in each column.
        """
        unique_values = dict()
        for col in self.data.columns:
            unique_value = self.data[col].nunique()
            unique_values[col] = unique_value
        return unique_values

    def print_shape(self):
        """
        Returns the shape of the DataFrame.

        Returns
        -------
        tuple
            A tuple containing the number of rows and columns in the DataFrame.
        """
        return self.data.shape

    def cal_null_percentage(self):
        """
        Returns the percentage of missing values in each column of the DataFrame.

        Returns
        -------
        pd.Series
            A Series containing the percentage of missing values in each column of the DataFrame.
        """
        null_percentage = self.data.isnull().mean() * 100
        return null_percentage

    def calculate_skewness(self):
        """
        Returns the skewness of the DataFrame's columns.

        Returns
        -------
        pd.Series
            A Series containing the skewness of the DataFrame's columns.
        """
        return self.data.skew()

    def get_data_info(self):
        """
        Writes a summary of the DataFrame's information and statistics to a text file.

        The summary includes data types, mean, median, standard deviation, number of unique values, shape,
        skewness, and percentage of missing values.

        Writes
        ------
        'data_information.txt'
            A text file containing the summary of the DataFrame's information and statistics.
        """
        output = (
            f'The datatype of each columns are as below: \n{self.columns_dtype()}\n\n'
            + '-' * 70 + '\n\n'
            + f'The mean of the data is: \n{self.stats_values()[0]} \n'
            + f'The median of the data is: \n{self.stats_values()[1]} \n'
            + f'The stddev of the data is: \n{self.stats_values()[2]}\n\n'
            + '-' * 70 + '\n\n'
            + f'The number of unique values in the columns are as below: \n{self.unique_values()}\n\n'
            + '-' * 70 + '\n\n'
            + f'The shape of the data is: \n{self.print_shape()}\n\n'
            + '-' * 70 + '\n\n'
            + f'The Skewness of the data is: \n{self.calculate_skewness()}\n\n'
            + '-' * 70 + '\n\n'
            + f'The percentage of the missing data as per each column are as below: \n{self.cal_null_percentage()}'
        )

        with open('data_information.txt', 'w') as f:
            f.write(output)


class Plotter(DataFrameInfo):
    """A class to plot various statistics and visualizations."""

    def __init__(self, data, folder_name):
        super().__init__(data)
        self.data = data
        self.folder_name = folder_name

    def plot_missing_values(self):
        """Plot bar chart for missing values."""
        null_percentage = self.cal_null_percentage()
        plt.bar(self.data.columns, null_percentage)
        plt.xticks(rotation=90)
        plt.ylabel('Percentage of missing values')
        plt.title('Missing Values')
        plt.savefig(f'images/{self.folder_name}/missing_value.png')
        plt.show()

    def plot_data_stats(self):
        """Plot bar charts for mean, median, and standard deviation."""
        mean_value, median_value, std_value = self.stats_values()

        plt.bar(mean_value.index, mean_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('Mean values')
        plt.title('Mean Values')
        plt.savefig(f'images/{self.folder_name}/mean_value.png')
        plt.show()

        plt.bar(median_value.index, median_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('Median values')
        plt.title('Median Values')
        plt.savefig(f'images/{self.folder_name}/median_value.png')
        plt.show()

        plt.bar(std_value.index, std_value.values)
        plt.xticks(rotation=90)
        plt.ylabel('Standard deviation values')
        plt.title('Standard Deviation')
        plt.savefig(f'images/{self.folder_name}/std_value.png')
        plt.show()

    def plot_skewness(self):
        """Plot bar chart for skewness."""
        skewness = self.data.skew()
        plt.bar(skewness.index, skewness.values)
        plt.xticks(rotation=90)
        plt.ylabel('Skewness values')
        plt.title('Skewness')
        plt.savefig(f'images/{self.folder_name}/skewness_value.png')
        plt.show()

    def skewness_histogram(self):
        """Plot histogram for skewness."""
        self.data.hist(figsize=(15, 12))
        plt.xticks(rotation=90)
        plt.title('Skewness')
        plt.savefig(f'images/{self.folder_name}/skewness_histogram.png')
        plt.show()

    def outlier_detection_boxplot(self):
        """Plot boxplots for outlier detection."""
        fig, ax = plt.subplots(2, 3, figsize=(15, 12))
        ax = ax.flatten()
        for i, col in enumerate(self.data.select_dtypes(include=[np.number]).columns):
            sns.boxplot(y=self.data[col], ax=ax[i])
            ax[i].set_title(col)
            ax[i].set_ylabel('')
            if i == 3:
                break
        plt.tight_layout()
        plt.show()
        fig.savefig(f'images/{self.folder_name}/boxplots.png')

    def correlation_matrix(self):
        """Plot heatmap for correlation matrix."""
        corr = self.data.corr()
        cmap = sns.color_palette("viridis", as_cmap=True)

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, square=True, linewidths=.5, annot=True, cmap=cmap, fmt=".2f")
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.savefig(f'images/{self.folder_name}/correlation.png')
        plt.show()

    def save_and_show_plots(self):
        """Save and display all plots."""
        self.plot_missing_values()
        self.plot_data_stats()
        self.plot_skewness()
        self.skewness_histogram()
        self.outlier_detection_boxplot()
        self.correlation_matrix()

class DataFrameTransform(DataFrameInfo):
    """A class for data transformation."""

    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def drop_unused_column(self):
        """Drop columns where all values are unique."""
        unique_values = self.unique_values()
        for col, count in unique_values.items():
            if count == self.data.shape[0]:
                self.data.drop(col, axis=1, inplace=True)

    def handle_null_values(self):
        """Handle missing values by dropping or imputing."""
        null_percentage = self.data.isnull().mean() * 100
        for col, value in zip(null_percentage.index, null_percentage.values):
            if value > 0:
                if self.data[col].dtype == 'O':
                    if value < 4.0:
                        self.data.drop(col, axis=1, inplace=True)
                    else:
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    if value < 4.0:
                        self.data.drop(col, axis=1, inplace=True)
                    else:
                        self.data[col].fillna(self.data[col].median(), inplace=True)

    def log_transform(self):
        """Apply log transformation to numerical columns."""
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.data[col] = np.log1p(self.data[col])

    def box_cox_transform(self, column):
        """Apply Box-Cox transformation to a specific column."""
        boxcox_column = stats.boxcox(self.data[column])
        boxcox_column = pd.Series(boxcox_column[0])
        return boxcox_column

    def yeo_johnson_transform(self, column):
        """Apply Yeo-Johnson transformation to a specific column."""
        yjh_column = stats.yeojohnson(self.data[column])
        yjh_column = pd.Series(yjh_column[0])
        return yjh_column

    def transform_data(self):
        """Transform data by applying transformations."""
        unique_values = self.unique_values()
        for col, count in unique_values.items():
            if count > 2 and self.data[col].dtype != 'O':
                self.data[col] = self.yeo_johnson_transform(col)

    def remove_outlier(self):
        """Remove outliers using IQR method."""
        df_clean = self.data.copy()
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
        return df_clean
    

# Main code block
if __name__ == '__main__':
    # Example usage and workflow
    file_path = 'data/data.csv'
    data_transformer = DataTransform(file_path=file_path)
    transformed_data = data_transformer.transform_columns()

    plotter = Plotter(transformed_data, 'pre_transformation')
    plotter.save_and_show_plots()

    data_informer = DataFrameInfo(transformed_data)
    data_informer.get_data_info()

    dataframe_transformer = DataFrameTransform(transformed_data)
    dataframe_transformer.drop_unused_column()
    dataframe_transformer.handle_null_values()
    dataframe_transformer.log_transform()

    # clean_data = dataframe_transformer.remove_outlier()

    plotter = Plotter(transformed_data, 'post_transformation')
    plotter.save_and_show_plots()

    save_to_csv(transformed_data, 'data/clean_data.csv')  






                
    








        




 
        










                