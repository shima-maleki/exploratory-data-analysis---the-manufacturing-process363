import yaml
import pandas as pd
from sqlalchemy import create_engine

def load_yaml(file_name):
    with open (file_name, 'r') as file:
        return yaml.safe_load(file)
    
class RDSDatabaseConnector:
    def __init__(self, config_data ):
        self.config_data = config_data
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'

    def create_engine(self):
        connection_string = f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.config_data['RDS_USER']}:{self.config_data['RDS_PASSWORD']}@{self.config_data['RDS_HOST']}:{self.config_data['RDS_PORT']}/{self.config_data['RDS_DATABASE']}"
        return create_engine(connection_string)
    
    def query_to_dataframe(self, query):
        with self.create_engine().connect() as connection:
            df = pd.read_sql(query, connection)
        return df

    def get_failure_data(self):
        query = "SELECT * FROM failure_data"
        return self.query_to_dataframe(query)
    
def save_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)


 # Replace with your actual database credentials
db_connector = RDSDatabaseConnector(config_data = load_yaml("credentials.yaml"))

failure_data_df = db_connector.get_failure_data()
save_to_csv(failure_data_df, 'data/data.csv')

print(failure_data_df)

