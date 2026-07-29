#!/usr/bin/env python3
"""
Parse downloaded TCGA-BRCA XML files into a single CSV.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/tcga_raw")
OUTPUT = Path("data/tcga/tcga_prepared.csv")


def parse_xml(filepath: Path) -> dict:
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Auto-detect namespace
    ns = {}
    if root.tag.startswith('{'):
        uri = root.tag.split('}')[0].strip('{')
        ns = {'ns': uri}
        patient_tag = 'ns:patient'
    else:
        patient_tag = 'patient'
    
    patient = root.find(f'.//{patient_tag}', ns)
    if patient is None:
        # Try without namespace
        patient = root.find('.//patient')
        if patient is None:
            return {}
    
    def get(tag):
        # Try with namespace first, then without
        for prefix in [f'ns:{tag}', tag]:
            el = patient.find(prefix, ns) if 'ns:' in prefix else patient.find(prefix)
            if el is not None and el.text:
                return el.text.strip()
        return None
    
    record = {
        'Patient_ID': get('bcr_patient_barcode'),
        'Age_at_diagnosis': get('age_at_initial_pathologic_diagnosis'),
        'Tumor_Stage': get('ajcc_pathologic_tumor_stage'),
        'Neoplasm_Histologic_Grade': get('neoplasm_histologic_grade'),
        'ER_Status': get('breast_carcinoma_estrogen_receptor_status'),
        'PR_Status': get('breast_carcinoma_progesterone_receptor_status'),
        'HER2_Status': get('lab_proc_her2_neu_immunohistochemistry_receptor_status'),
        'Overall_Survival_Status': get('vital_status'),
        'Overall_Survival_Months': get('days_to_last_followup'),
    }
    
    # Debug: print first successful parse
    if record['Patient_ID'] and not hasattr(parse_xml, '_printed'):
        print(f"Sample from {filepath.name}: {record}")
        parse_xml._printed = True
    
    return record


# Parse all XMLs
records = []
for xml_file in sorted(DATA_DIR.glob("*.xml"))[:50]:  # Test first 50
    try:
        record = parse_xml(xml_file)
        if record.get('Patient_ID'):
            records.append(record)
    except Exception as e:
        print(f"Failed {xml_file.name}: {e}")

print(f"Parsed {len(records)} patients from {len(list(DATA_DIR.glob('*.xml')))} files")

if not records:
    print("No records parsed. XML structure may differ.")
    # Print raw tag of first file for debugging
    first = next(DATA_DIR.glob("*.xml"))
    tree = ET.parse(first)
    print(f"Root tag: {tree.getroot().tag}")
    exit(1)

df = pd.DataFrame(records)

# Clean up
df['Age_at_diagnosis'] = pd.to_numeric(df['Age_at_diagnosis'], errors='coerce')
df['Overall_Survival_Months'] = pd.to_numeric(df['Overall_Survival_Months'], errors='coerce') / 30.44

# Stage mapping
stage_map = {
    'Stage IA': 'I', 'Stage IB': 'I', 'Stage I': 'I',
    'Stage IIA': 'II', 'Stage IIB': 'II', 'Stage II': 'II',
    'Stage IIIA': 'III', 'Stage IIIB': 'III', 'Stage IIIC': 'III', 'Stage III': 'III',
    'Stage IV': 'IV',
}
df['Tumor_Stage'] = df['Tumor_Stage'].map(stage_map)

# Receptor status
for col in ['ER_Status', 'PR_Status', 'HER2_Status']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.title()

# Survival status
if 'Overall_Survival_Status' in df.columns:
    df['Overall_Survival_Status'] = df['Overall_Survival_Status'].astype(str).map({
        'Alive': 'Living', 'Dead': 'Deceased'
    })

# Menopausal state
df['Inferred_Menopausal_State'] = df['Age_at_diagnosis'].apply(
    lambda x: 'Post' if pd.notna(x) and x >= 50 else 'Pre'
)

# Missing fields (TCGA XML doesn't have these)
for col in ['Tumor_Size', 'Lymph_nodes_examined_positive', 
            'Relapse_Free_Status', 'Relapse_Free_Months',
            'Chemotherapy', 'Hormone_Therapy', 'Radio_Therapy']:
    df[col] = None

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} patients to {OUTPUT}")
print(df.head())