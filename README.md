# Urban Hazard Detector

A Python tool for detecting buildings, amenities, and sports infrastructures located on or near risky natural features such as wetlands, dunes, coastlines, or rivers/streams.

## Features

- Downloads OpenStreetMap data for specified locations
- Identifies structures near natural hazards (wetlands, dunes, coastlines, rivers)
- Filters out artificial water sources (pools, small artificial lakes)
- Provides geocoded addresses for identified structures
- Exports results to CSV format

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

The script will analyze the Nessebar Municipality area in Bulgaria by default and generate a `danger_summary_exact.csv` file with the results.

## Configuration

You can modify the following parameters in `main.py`:
- `BUFFER_METERS`: Buffer distance for hazard detection (default: 20m)
- `MIN_WATER_AREA`: Minimum water body area to consider natural (default: 50 m²)
- Location and bounding box for analysis

## Output

The tool generates `danger_summary_exact.csv` containing:
- Type of structure (Buildings/Amenities/Sports)
- Name of the structure
- Address
- Latitude and Longitude
- Cause of risk (nearby hazard type)

## Requirements

See `requirements.txt` for all dependencies.

