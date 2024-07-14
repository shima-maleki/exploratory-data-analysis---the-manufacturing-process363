import yaml
import pandas as pd
from sqlalchemy import create_engine

def load_yaml(file_name):
    """Load YAML file and return its contents as a dictionary."""
    with open(file_name, 'r') as file:
        return yaml.safe_load(file)

class RDSDatabaseConnector:
    """A class to connect to an RDS PostgreSQL database and perform operations."""

    def __init__(self, config_data):
        """
        Initialize with database connection details.

        Parameters:
        - config_data (dict): Dictionary containing RDS connection parameters
        """
        self.config_data = config_data
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'

    def create_engine(self):
        """
        Create SQLAlchemy engine for database connection.

        Returns:
        - SQLAlchemy Engine object
        """
        connection_string = f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.config_data['RDS_USER']}:{self.config_data['RDS_PASSWORD']}@{self.config_data['RDS_HOST']}:{self.config_data['RDS_PORT']}/{self.config_data['RDS_DATABASE']}"
        return create_engine(connection_string)

    def query_to_dataframe(self, query):
        """
        Execute SQL query and return results as a pandas DataFrame.

        Parameters:
        - query (str): SQL query string

        Returns:
        - pandas DataFrame
        """
        try:
            with self.create_engine().connect() as connection:
                df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None

    def get_failure_data(self):
        """
        Fetch all data from the 'failure_data' table.

        Returns:
        - pandas DataFrame with fetched data
        """
        query = "SELECT * FROM failure_data"
        return self.query_to_dataframe(query)

def save_to_csv(dataframe, file_path):
    """
    Save pandas DataFrame to CSV file.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame to be saved
    - file_path (str): File path to save the CSV file
    """
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")

if __name__ == '__main__':
    # Load database credentials from YAML file
    try:
        db_credentials = load_yaml("credentials.yaml")
    except FileNotFoundError:
        print("Error: credentials.yaml file not found.")
        exit()

    # Initialize database connector with credentials
    db_connector = RDSDatabaseConnector(config_data=db_credentials)

    # Fetch data from database
    failure_data_df = db_connector.get_failure_data()

    if failure_data_df is not None:
        # Save fetched data to CSV
        save_to_csv(failure_data_df, 'data/data.csv')

        # Print fetched data frame
        print(failure_data_df)
    else:
        print("Failed to fetch data from database.")
