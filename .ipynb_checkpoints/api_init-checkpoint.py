import os
from dotenv import load_dotenv
from fredapi import Fred
import eikon as ek

# Only works if you saved the keys in the machine as environment variables
def initialize_apis():
    load_dotenv()
    
    fred_api_key = os.getenv('FRED_API_KEY')
    fred = Fred(fred_api_key)
    
    eikon_api_key = os.getenv('EIKON_API_KEY')
    ek.set_app_key(eikon_api_key)

    return (fred, ek)
