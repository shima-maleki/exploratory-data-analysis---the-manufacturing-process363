# Exploratory Data Analysis: The Manufacturing Process

## Project Description
This project focuses on performing exploratory data analysis (EDA) on a dataset from a manufacturing process. The aim is to understand the operating ranges of various machine settings, investigate the causes of machine failures, and identify potential risk factors that lead to these failures. Through this analysis, we aim to provide insights that can help optimize machine settings and reduce the likelihood of failures in the manufacturing process.

## Key Objectives:
1. Understand the operating ranges for different machine settings (e.g., air temperature, process temperature, rotational speed, torque, and tool wear).
2. Determine the number and leading causes of machine failures.
3. Investigate correlations between machine settings and types of failures.
4. Develop strategies to mitigate machine failures based on identified risk factors.

## What We Learned:
1. How to perform EDA on a manufacturing dataset.
2. Techniques to visualize data distributions and correlations.
3. Methods to identify potential risk factors for machine failures.
4. Strategies for optimizing machine settings to prevent failures.

## Installation Instructions
To run this project, you will need Python and several libraries. Follow the steps below to set up your environment:

1. Clone the Repository: https://github.com/shima-maleki/exploratory-data-analysis---the-manufacturing-process363.git
2. cd exploratory-data-analysis---the-manufacturing-process363
3. Create a Virtual Environment:

```
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

4. Install Dependencies:

```
pip install -r requirements.txt
```

## Usage Instructions
Activate the Virtual Environment:

```
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

## Run the Analysis:
1. for detail EDA and Data transformation

```
python eda.py
```

2. Open the Jupyter notebook provided in the repository and execute the cells step-by-step to perform the EDA and visualize the results.

## View Results:
The analysis includes visualizations and summary statistics to help you understand the operating ranges and failure causes in the manufacturing process.

## File Structure of the Project

```
manufacturing-process-eda/
├── data/
│   └── clean_data.csv
│   └── data.csv
├── images/
│   └── pre_transformation
│   └── post_transformation
├── analysis_and_visualisation.ipynb
├── data_preprocessing.ipynb
├── db_utils.py
├── eda.py
├── README.md  # Project description and instructions
├── requirements.txt  # List of dependencies
└── LICENSE  # License information
```

## Contact
For any questions or feedback, please reach out to:

Name: Shima Maleki
Email: Shimamaleki95@yahoo.com


## License Information
This project is licensed under the MIT License. See the LICENSE file for more details.