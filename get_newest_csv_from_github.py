# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:03:05 2023

@author: codyt
"""

import requests
import os
import datetime

now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

# Define the GitHub repository URL and file path
repository_url = 'https://github.com/cwilsonCrystalSteel/Python'



files = ['archive_FED.csv','Archive_CSM.csv','archive_CSF.csv','production_worksheet.csv']

for file in files:
    
    github_file_path = 'Speedo_Dashboard/' + file
    
    # Construct the raw file URL
    raw_url = f'{repository_url}/raw/main/{github_file_path}'
    
    # Make an HTTP GET request to download the file
    response = requests.get(raw_url)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Specify the local file path where you want to save the downloaded file
        local_dir = ".\\data\\"
        local_file_path = local_dir + file
        
        # rename the old file if it exists so that the new one doesnt overwrite
        if os.path.exists(local_file_path):
            new_file_name = local_dir + now_str + file
            os.rename(local_file_path, new_file_name)
            
            
        # Write the content of the response to the local file
        with open(local_file_path, 'wb') as newfile:
            newfile.write(response.content)
                
            
        
        print(f'File downloaded and saved as {local_file_path}')
    else:
        print(f'Failed to download the file. HTTP Status Code: {response.status_code}')
