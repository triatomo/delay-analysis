"""
Cleaning raw data
"""

import pandas as pd
pd.set_option('display.max_columns', None)

raw_data = pd.read_csv('csv cleaning/duplicates_removed.csv', sep=';')

cleaned_data = raw_data.dropna(subset=['Carrier Company & Country', 'Shipment Planned Pickup Time', 'Shipment First Hub Scan at LMC (FHS) Time', 'Shipment First Delivery Attempt (FDA) Time', 'Shipment Delivery Time', 'Delivery from injection', 'FHS-FDA', 'PPU-FHS', 'FDA-Delivery'])
cleaned_data.to_csv("cleaned_data.csv", index=False, encoding='utf8')

# # Remove missing values in 4 columns
# missing_times_removed=data.dropna(subset=['Shipment Planned Pickup Time', 'Shipment First Hub Scan at LMC (FHS) Time', 'Shipment First Delivery Attempt (FDA) Time', 'Shipment Delivery Time'])
# missing_times_removed.to_csv("missing_times_removed.csv", index=False, encoding='utf8')

# # Remove rows that do not fill the requirement
# filled_req=missing_times_removed.dropna(subset=['Delivery from injection', 'FHS-FDA', 'PPU-FHS', 'FDA-Delivery'])
# filled_req.to_csv("filled_req.csv", index=False, encoding='utf8')
