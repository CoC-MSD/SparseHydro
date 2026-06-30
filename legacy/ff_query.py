from requests import Session
import requests.exceptions
from requests.auth import HTTPBasicAuth
from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter
from pathlib import Path

FF_USERNAME = "rest_opt_data"
FF_PASSWORD = "P@55w0rd456!"
types_and_titles = {"Flow":"flow_list.tsf",
                    "Level":"lstDepthTs.tsf",
                    "Velocity":"lstVelocityTs.tsf",
                    "Rain":"rain_list.tsf",
                    "StormFlow":"lstStormTs.tsf",
                    "BaseFlow":"lstBaseflowTs.tsf",
                    "RecommendedEventFileForOptimization":"lstEventsImported Events 0.dat"}


def get_option_from_list(options):
    while True:
        print("\nPlease select an option:")
        # Print the options with 1-based indexing for the user
        for index, item in enumerate(options, start=1):
            print(f"{index}. {item}")
            
        try:
            choice = int(input("\nEnter the number of your choice: "))
            
            # Validate if the number matches an item in the list
            if 1 <= choice <= len(options):
                selected_item = options[choice - 1] # Convert back to 0-based index
                print(f"You selected: {selected_item}")
                return choice-1
            else:
                print("Invalid number. Please pick a number from the list.")
                
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def download_file(name, url, output_folder):

    try:
        # 1. Make the GET request
        response = s.get(url, auth=HTTPBasicAuth(FF_USERNAME, FF_PASSWORD), headers={'Host': 'msdgc.org'})

        # 2. Raise an exception for bad HTTP status codes (4xx or 5xx)
        response.raise_for_status()
        
        # 3. Parse the JSON response into a Python dict/list
        #data = response.json()
        # Open file with 'wb' to write raw bytes
        output_path = Path(output_folder) / f"{name}"
        with open(output_path, "wb") as file:
            file.write(response.content)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.JSONDecodeError:
        print("Error: Response content was not valid JSON.")
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")

    return output_path

s = Session()
s.mount('https://', HostHeaderSSLAdapter())

# This query gives us a list of possible flow meter data titles.
url = "https://flowfinity.msdgc.org/Flowfinity/rest/ModelOpt/DataSets/FStandardizedView"

try:
    # 1. Make the GET request
    response = s.get(url, auth=HTTPBasicAuth(FF_USERNAME, FF_PASSWORD), headers={'Host': 'msdgc.org'})

    # 2. Raise an exception for bad HTTP status codes (4xx or 5xx)
    response.raise_for_status()
    
    # 3. Parse the JSON response into a Python dict/list
    data = response.json()

    # 4. Safely access dictionary elements
    # Extract just the names into a clean list
    data_titles = [title["DataTitle"] for title in data]

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.JSONDecodeError:
    print("Error: Response content was not valid JSON.")
except requests.exceptions.RequestException as err:
    print(f"An error occurred: {err}")




selected_data_index = get_option_from_list(data_titles)

# print(list(filter(lambda x:x["DataTitle"]==data_titles[selected_data_index],data)))


base_url = "https://flowfinity.msdgc.org/Flowfinity/rest/ModelOpt/DataSets/ViewStandardRecord"

downloaded_files = []

for type, title in types_and_titles.items():
    try:
        type_url = data[selected_data_index][type]["URL"]
        path = download_file(title, base_url+type_url, "downloads")
        downloaded_files.append(path)
    except KeyError as key_err:
        print(f"Did not find: {key_err}")

print("Wrote: ",*downloaded_files, sep="\n")




# https://flowfinity.msdgc.org/Flowfinity/rest/ModelOpt/DataSets/ViewStandardRecord



