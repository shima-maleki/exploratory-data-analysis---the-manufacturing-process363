import yaml

def load_yaml(file_name):
    with open (file_name, 'r') as file:
        return yaml.safe_load(file)
    
class RDSDatabaseConnector:
    pass

print(load_yaml('credentials.yaml'))