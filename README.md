
# Home-credit-default-risk

This is a repository to share the analysis of the home-credit-default-risk from kaggle.com (https://www.kaggle.com/c/home-credit-default-risk).



# Directory structure
    
   This repository was structured as below:
    
- 1-script

    This folder contain all script file that will be used to discover new insight from this project.

- 2-data

    This folder contain the data file used on analysis, all files was downloaded from 
    __"https://www.kaggle.com/c/home-credit-default-risk/data"__. Each new column on the files was create to improve 
    analysis, all the new columns is available on file "HomeCredit_columns_description".

- 3-logs
    
    This folfer was created to store all logs o execution, this is userfull to identify the possible performance problem.

- 4-config

    This folder was created to store all configurable parameter that will be exposed.
    This directory will also contain environment settings that are required for all process analysis.
    

# Environment

To make more easy to reproduce the result presente here, all python libraries used will be available at __"4-config/config_linux_env.yml"__ and could be used to create a new conda env.

- Linux environment
```
    conda env create -f config_linux_env.yml
    conda source activate hcdr-project
```
    
After create a new conda enviroment it's necessary create a new ipkernel, juste follow the next step bellow to do that.
    
- Start conda enviroment
```
    source activate hcdr-project
```
   
- Install ipykernel on conda enviroment   
```
    conda install ipykernel
``` 

- Create a new ipkernel.
```
    python -m ipykernel install --user --name hcdr-project --display-name hcdr-project
```





