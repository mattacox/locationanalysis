# --- Imports ---

from censusdis.datasets import ACS5
from censusdis import states
import censusdis.data as ced
import constants
import time

def safe_download_acs(vintage, max_retries=3, delay=3):
    # -- Download ACS data with retries to avoid hard script failure ---
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: pulling ACS {vintage}")

            # --- define the main download parameters for Census API pull via censusdis        
            data = ced.download(
                dataset=ACS5,
                vintage=vintage,
                download_variables=constants.bg_vars,
                state=states.NC,
                county=['077'],
                block_group='*',
                with_geometry=True,
            )
            return data
        except Exception as e:
            print(f"Error pulling ACS {vintage}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay} sec...")
                time.sleep(delay)
            else:
                print(f"Failed after {max_retries} attempts — skipping {vintage}")
                return None


