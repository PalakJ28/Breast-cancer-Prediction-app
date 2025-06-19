
# Selected features for the model
SELECTED_FEATURES = [
    'Age recode with <1 year olds',
    'Race recode (White, Black, Other)',
    'Sex',
    'Year of diagnosis',
    'Primary Site',
    'Histologic Type ICD-O-3',
    'Grade Recode (thru 2017)',
    'Combined Summary Stage (2004+)',
    'Regional nodes examined (1988+)',
    'Regional nodes positive (1988+)',
    'Tumor Size Summary (2016+)',
    'Radiation recode',
    'Chemotherapy recode (yes, no/unk)',
    'RX Summ--Surg Prim Site (1998+)',
    'Marital status at diagnosis',
    'Median household income inflation adj to 2022',
    'Rural-Urban Continuum Code',
    'SEER cause-specific death classification'
]

# Target column
TARGET_COLUMN = 'SEER cause-specific death classification' 