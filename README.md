# openFDA-DeviceEvent
Exploration of openFDA adverse events database for medical devices
## Plan
In this project, we are looking for device adverse events. We are particularly interested in:
    - Cause of failure
    - Date of failure (ie age of device)
We want to see which is the most common cause for given device to fail and at which stage of use this occurs.
Relevant fields could be (for full reference see https://open.fda.gov/device/event/reference/):
### Event
device_date_of_manufacturer
date_of_event, date_report, date_received
previous_use_code, remedial_action
single_use_flag
### Source
reprocessed_and_reused_flag
### Device
device.generic_name
device.expiration_date_of_device, device.device_age_text
device.implant_flag, device.date_removed_flag
device.manufacturer_d_name, device.manufacturer_d_state, device.manufacturer_d_country
### Patient
patient.sequence_number_outcome, patient.sequence_number_treatment
### Report Text
mdr_text.text, mdr_text.text_type_code
### Reporter Dependent Fields

#### By user facility / importer
report_date
event_location
manufacturer_name, manufacturer_country
manufacturer_g1_name, manufacturer_g1_state
### OpenFDA fields
device_class
### Further interesting fields:
Source: reporter_occupation_code
Device: device.device_operator
