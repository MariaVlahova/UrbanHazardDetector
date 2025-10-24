"""
Sunny Beach Environmental Risk Detection
----------------------------------------
Detects buildings, amenities, and sports infrastructures located
on or near risky natural features such as wetlands, dunes,
coastlines, or rivers/streams.

Excludes artificial water sources (pools, small artificial lakes)
and very small water bodies (<50 m²).

Outputs: danger_summary_exact.csv
"""

# -----------------------------
# Imports
# -----------------------------
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.strtree import STRtree
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm
import pandas as pd
import time
import logging
import sys
from typing import Optional, List, Tuple

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_LOCATION = "Sunny Beach, Bulgaria"
DEFAULT_BBOX = (42.710, 42.670, 27.750, 27.680)  # north, south, east, west for Sunny Beach
BUFFER_METERS = 20  # Buffer distance in meters
MIN_WATER_AREA = 50  # m² - minimum area to consider a water body natural
METRIC_CRS = "EPSG:32635"  # UTM Zone 35N for Bulgaria

# -----------------------------
# Utility functions
# -----------------------------
def get_layer(tags: dict, bbox_geom, max_retries: int = 3) -> gpd.GeoDataFrame:
    """Download OSM features by tags within the bounding box with retry logic."""
    for attempt in range(max_retries):
        try:
            gdf = ox.features.features_from_polygon(bbox_geom, tags=tags)
            if not gdf.empty:
                gdf["geometry"] = gdf.geometry.apply(make_valid)
                gdf = gdf.to_crs(epsg=4326)

            return gdf
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to download layer: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download layer after {max_retries} attempts: {tags}")
                return gpd.GeoDataFrame()  # Return empty GeoDataFrame on failure
    return gpd.GeoDataFrame()


def reverse_geocode(geometry, geolocator, max_retries: int = 3) -> Optional[str]:
    """Safely reverse-geocode a geometry centroid to an address with retry logic."""
    lat, lon = geometry.centroid.y, geometry.centroid.x
    
    for attempt in range(max_retries):
        try:
            location = geolocator.reverse((lat, lon), timeout=10)
            return location.address if location else None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < max_retries - 1:
                logger.debug(f"Geocoding attempt {attempt + 1}/{max_retries} failed, retrying...")
                time.sleep(1)
            else:
                logger.warning(f"Geocoding failed after {max_retries} attempts for ({lat}, {lon}): {e}")
                return None
        except Exception as e:
            logger.warning(f"Unexpected geocoding error for ({lat}, {lon}): {e}")
            return None
    return None


def get_location_bbox(location_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box for a location name using geocoding."""
    try:
        geolocator = Nominatim(user_agent="urban_hazard_detector")
        location = geolocator.geocode(location_name, timeout=10)
        if location and location.raw.get('boundingbox'):
            bbox = location.raw['boundingbox']
            # Nominatim returns [south, north, west, east]
            south, north, west, east = map(float, bbox)
            logger.info(f"Found location: {location.address}")
            logger.info(f"Bounding box: N={north}, S={south}, E={east}, W={west}")
            return (north, south, east, west)
        else:
            logger.error(f"Could not find bounding box for location: {location_name}")
            return None
    except Exception as e:
        logger.error(f"Error geocoding location '{location_name}': {e}")
        return None


def find_exact_causes(obj_geom, gdf_natural, gdf_water, buffer_meters: float = BUFFER_METERS) -> List[str]:
    """Return a list of all nearby natural or waterway causes (with names if available).
    
    Uses spatial indexing for efficient proximity search.
    """
    causes = []
    
    # Convert buffer from meters to degrees (approximate at mid-latitude)
    # More accurate would be to use a projected CRS
    buffer_deg = buffer_meters / 111000  # Rough approximation

    # --- Nearby waterways ---
    if not gdf_water.empty:
        for _, w in gdf_water.iterrows():
            d = obj_geom.distance(w.geometry)
            if d < buffer_deg:
                w_type = w.get("waterway", "waterway")
                w_name = w.get("name")
                cause = f"Near {w_type} ({w_name})" if w_name else f"Near {w_type}"
                causes.append(cause)

    # --- Nearby natural features ---
    if not gdf_natural.empty:
        for _, n in gdf_natural.iterrows():
            d = obj_geom.distance(n.geometry)
            if d < buffer_deg:
                n_type = n.get("natural", "natural area")
                n_name = n.get("name")
                cause = f"On {n_type} ({n_name})" if n_name else f"On {n_type}"
                causes.append(cause)

    if not causes:
        causes = ["Unknown"]

    return causes


def find_structures_near_hazards_optimized(gdf_structures: gpd.GeoDataFrame, 
                                           gdf_hazards: gpd.GeoDataFrame, 
                                           buffer_meters: float) -> gpd.GeoDataFrame:
    """Use spatial indexing (STRtree) to efficiently find structures near hazards."""
    if gdf_structures.empty or gdf_hazards.empty:
        return gdf_structures.head(0)
    
    # Convert buffer from meters to degrees (approximate)
    buffer_deg = buffer_meters / 111000
    
    # Create spatial index for hazards
    hazard_geoms = list(gdf_hazards.geometry)
    tree = STRtree(hazard_geoms)
    
    # Find structures near any hazard
    dangerous_indices = []
    for idx, struct_geom in enumerate(gdf_structures.geometry):
        # Query nearby hazards
        nearby_indices = tree.query(struct_geom.buffer(buffer_deg))
        if len(nearby_indices) > 0:
            # Check if any are actually within buffer distance
            for hazard_idx in nearby_indices:
                if struct_geom.distance(hazard_geoms[hazard_idx]) < buffer_deg:
                    dangerous_indices.append(idx)
                    break
    
    return gdf_structures.iloc[dangerous_indices]

# -----------------------------
# Main workflow
# -----------------------------
def run_risk_detection(location_name: Optional[str] = None, bbox: Optional[Tuple[float, float, float, float]] = None):
    """Run risk detection analysis for a given location or bounding box.
    
    Args:
        location_name: Name of location to analyze (e.g., "Sunny Beach, Bulgaria")
        bbox: Optional manual bounding box as (north, south, east, west)
    """
    # Determine bounding box
    if bbox:
        north, south, east, west = bbox
        logger.info(f"Using provided bounding box: N={north}, S={south}, E={east}, W={west}")
    elif location_name:
        bbox_result = get_location_bbox(location_name)
        if not bbox_result:
            logger.error("Could not determine bounding box. Exiting.")
            return
        north, south, east, west = bbox_result
    else:
        north, south, east, west = DEFAULT_BBOX
        logger.info(f"Using default location: {DEFAULT_LOCATION}")
    
    bbox_geom = box(west, south, east, north)

    logger.info("⬇️  Downloading OSM layers...")

    # Structures
    tags_buildings = {"building": True, "building:part": True}
    tags_amenities = {
        "amenity": [
            "school", "bank", "hospital", "clinic", "police",
            "fire_station", "townhall", "courthouse", "library",
            "university", "college"
        ],
        "office": ["government", "administration"],
        "tourism": ["hotel", "apartment", "hostel", "guest_house"]
    }
    tags_sports = {
        "leisure": ["stadium", "sports_centre", "pitch",
                    "swimming_pool", "track", "playground"]
    }

    gdf_buildings = get_layer(tags_buildings, bbox_geom)
    gdf_amenities = get_layer(tags_amenities, bbox_geom)
    gdf_sports = get_layer(tags_sports, bbox_geom)

    # Natural + Water features
    tags_natural = {"natural": ["wetland", "water", "coastline", "dune"]}
    tags_water = {"waterway": ["river", "stream", "canal", "drain", "ditch"]}

    gdf_natural = get_layer(tags_natural, bbox_geom)
    gdf_water = get_layer(tags_water, bbox_geom)

    # -----------------------------
    # Exclude artificial water sources
    # -----------------------------
    if not gdf_natural.empty:
        # Remove swimming pools only if 'leisure' column exists
        if "leisure" in gdf_natural.columns:
            gdf_natural = gdf_natural[~gdf_natural["leisure"].fillna("").isin(["swimming_pool"])]

        # Remove very small water bodies (<50 m²) - convert to metric CRS first
        gdf_natural_metric = gdf_natural.to_crs(METRIC_CRS)
        gdf_natural_metric["area_m2"] = gdf_natural_metric.geometry.area
        valid_indices = gdf_natural_metric[gdf_natural_metric["area_m2"] >= MIN_WATER_AREA].index
        gdf_natural = gdf_natural.loc[valid_indices]
        logger.info(f"Filtered out {len(gdf_natural_metric) - len(gdf_natural)} small water bodies (<{MIN_WATER_AREA} m²)")

    logger.info("✅ Layers downloaded and filtered.")
    logger.info(f"Buildings: {len(gdf_buildings)}, Amenities: {len(gdf_amenities)}, Sports: {len(gdf_sports)}")
    logger.info(f"Natural (filtered): {len(gdf_natural)}, Waterways: {len(gdf_water)}")

    all_structures = {
        "Buildings": gdf_buildings,
        "Amenities": gdf_amenities,
        "Sports": gdf_sports
    }

    danger_results = {}

    # Identify risky structures using optimized spatial indexing
    logger.info("⚙️  Detecting dangerous structures...")
    for name, gdf in all_structures.items():
        if gdf.empty:
            continue
        
        # Use optimized spatial indexing instead of unary_union
        danger_nat = find_structures_near_hazards_optimized(gdf, gdf_natural, BUFFER_METERS)
        danger_water = find_structures_near_hazards_optimized(gdf, gdf_water, BUFFER_METERS)

        danger_results[f"{name}_natural"] = danger_nat
        danger_results[f"{name}_waterway"] = danger_water
        
        logger.info(f"{name}: {len(danger_nat)} near natural features, {len(danger_water)} near waterways")

    # Reverse geocode
    logger.info("🌍 Reverse geocoding...")
    geolocator = Nominatim(user_agent="urban_hazard_detector")
    tqdm.pandas()
    for key, gdf in danger_results.items():
        if not gdf.empty:
            logger.info(f"Geocoding {len(gdf)} locations for {key}...")
            gdf["address"] = gdf.geometry.progress_apply(lambda g: reverse_geocode(g, geolocator))
            danger_results[key] = gdf
            time.sleep(1)  # Rate limiting between batches

    # Summarize results
    summary_rows = []
    for layer_name, gdf in danger_results.items():
        if gdf.empty:
            continue

        for _, row in gdf.iterrows():
            name = (
                row.get("name")
                or row.get("amenity")
                or row.get("building")
                or row.get("leisure")
                or "Unnamed"
            )
            address = row.get("address", "No address")
            causes = find_exact_causes(row.geometry, gdf_natural, gdf_water)
            lon, lat = row.geometry.centroid.x, row.geometry.centroid.y

            summary_rows.append({
                "type": layer_name.split("_")[0],
                "name": name,
                "address": address,
                "latitude": lat,
                "longitude": lon,
                "cause": ", ".join(causes)  # CSV-friendly
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.drop_duplicates(subset=['address', 'name'], keep='first')
    df_summary.to_csv("danger_summary_exact.csv", index=False, encoding="utf-8-sig")

    logger.info(f"✅ Saved {len(df_summary)} entries to danger_summary_exact.csv")
    logger.info(f"\nSample results:\n{df_summary.head()}")
    
    return df_summary

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Fetch Nessebar Municipality and extract bounding box
    bbox = None
    
    try:
        gdf_nessebar = ox.geocode_to_gdf("Nessebar Municipality, Burgas, Bulgaria")
        bbox_geom = gdf_nessebar.geometry.iloc[0]
        
        # Extract bounding box from the polygon
        bounds = bbox_geom.bounds  # Returns (minx, miny, maxx, maxy)
        west, south, east, north = bounds
        bbox = (north, south, east, west)  # Format expected by run_risk_detection
        
        logger.info("✅ Using Nessebar Municipality polygon")
        logger.info(f"Bounding box: N={north}, S={south}, E={east}, W={west}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch municipality polygon: {e}")
        north, south, east, west = 42.7485666, 42.6450690, 27.9065768, 27.6385
        bbox = (north, south, east, west)
        logger.info("⚠️ Using fallback bounding box")

    
    # Run risk detection with the bounding box
    run_risk_detection(bbox=bbox)
