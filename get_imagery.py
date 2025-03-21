#!/usr/bin/env python3
"""
Historical Satellite Imagery Fetcher
-----------------------------------
This script fetches historical satellite imagery for a given location and date range.
Uses the latest Sentinel Hub API (sentinelhub-py client version 3.x).
"""

import os
import sys
import datetime
import argparse
import requests
from datetime import timedelta
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import geojson
from geopy.geocoders import Nominatim
from shapely.geometry import Point, box

# Import the latest sentinelhub package
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions,
    Geometry,
    SentinelHubCatalog
)

def setup_sentinel_config():
    """Set up the Sentinel Hub configuration."""
    config = SHConfig()
    
    # Check if credentials are set as environment variables
    client_id = os.environ.get('SENTINEL_CLIENT_ID')
    client_secret = os.environ.get('SENTINEL_CLIENT_SECRET')
    
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        
        # Set the OAuth URL
        config.sh_token_url = "https://services.sentinel-hub.com/oauth/token"
    else:
        print("Warning: Sentinel Hub credentials not found in environment variables.")
        print("Please set SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET or enter them now.")
        config.sh_client_id = input("Enter your Sentinel Hub client ID: ")
        config.sh_client_secret = input("Enter your Sentinel Hub client secret: ")
    
    # Test the configuration
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Sentinel Hub credentials are required.")
    
    return config

def address_to_coordinates(address):
    """Convert an address to latitude and longitude."""
    geolocator = Nominatim(user_agent="satellite_imagery_fetcher")
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not geocode address: {address}")
            return None
    except Exception as e:
        print(f"Error geocoding address: {e}")
        return None

def create_bounding_box(latitude, longitude, width_km=1.0, height_km=1.0):
    """Create a bounding box around the given coordinates."""
    # Approximate conversion from km to degrees
    # This is a simplification and will be less accurate at extreme latitudes
    lat_offset = height_km / 111.32  # 1 degree latitude is approximately 111.32 km
    lon_offset = width_km / (111.32 * np.cos(np.radians(latitude)))  # Longitude degrees vary with latitude
    
    min_lon = longitude - lon_offset / 2
    max_lon = longitude + lon_offset / 2
    min_lat = latitude - lat_offset / 2
    max_lat = latitude + lat_offset / 2
    
    # Create Sentinel Hub BBox object
    return BBox([min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

def get_available_dates(config, bbox, data_collection, start_date, end_date):
    """Get a list of dates with available imagery for the location and date range using the Catalog API."""
    # Create a catalog object
    catalog = SentinelHubCatalog(config=config)
    
    # Convert BBox to Geometry for the catalog search
    geometry = Geometry(bbox.geometry, bbox.crs)
    
    # Convert dates to ISO format
    start_date_iso = start_date.isoformat() + 'T00:00:00Z'
    end_date_iso = end_date.isoformat() + 'T23:59:59Z'
    
    
    # Map the data collection to the correct catalog collection
    collection_map = {
        DataCollection.SENTINEL2_L2A: "sentinel-2-l2a",
        DataCollection.SENTINEL2_L1C: "sentinel-2-l1c",
        DataCollection.HARMONIZED_LANDSAT_SENTINEL: "landsat-ot-l2"
    }
    
    collection = collection_map.get(data_collection)
    if not collection:
        raise ValueError(f"Unsupported data collection: {data_collection}")
    
    # Perform the search
    search_iterator = catalog.search(
        collection=collection,
        datetime=f"{start_date_iso}/{end_date_iso}",
        geometry=geometry,
        fields={
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover"
            ]
        }
    )
    
    # Process search results
    available_dates = []
    
    try:
        results = list(search_iterator)
        # print(f"All results: {results}")
        for item in results:
            try:
                # Extract date from the item properties
                date_str = item["properties"]["datetime"]
                date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                
                # Get cloud cover if available
                cloud_cover = item["properties"].get("eo:cloud_cover", None)
                
                available_dates.append({
                    "date": date,
                    "id": item["id"],
                    "cloud_cover": cloud_cover
                })
            except (KeyError, ValueError) as e:
                print(f"Error processing item: {e}")
                continue
    except Exception as e:
        print(f"Error searching catalog: {e}")
    
    # Sort and filter dates (e.g., by cloud cover)
    available_dates.sort(key=lambda x: x["date"])
    
    # Print summary
    print(f"Found {len(available_dates)} available images.")
    for item in available_dates[:5]:  # Show first 5 for preview
        cloud_info = f", Cloud cover: {item['cloud_cover']}%" if item['cloud_cover'] is not None else ""
        print(f"  - {item['date'].isoformat()}{cloud_info}")
    
    if len(available_dates) > 5:
        print(f"  ... and {len(available_dates) - 5} more")
    
    return available_dates

def fetch_sentinel_image(config, bbox, date_info, data_collection, output_path, resolution=60):
    """Fetch a satellite image for the given location and date."""
    try:
        # Calculate image dimensions based on bbox and desired resolution
        size = bbox_to_dimensions(bbox, resolution)
        # print(f"Image dimensions: {size}")  # Debugging line
        
        # Define the evalscript (true color RGB composite with cloud masking)
        evalscript = """
        //VERSION=3
        let minVal = 0.0;
        let maxVal = 0.4;

        let viz = new HighlightCompressVisualizer(minVal, maxVal);

        function setup() {
        return {
            input: ["B04", "B03", "B02","dataMask"],
            output: { bands: 4 }
        };
        }

        function evaluatePixel(samples) {
            let val = [samples.B04, samples.B03, samples.B02,samples.dataMask];
            return viz.processList(val);
        }
        """
        
        # Date range (use exact date for catalog items)
        date = date_info["date"]
        time_interval = f"{date.isoformat()}"
        
        # Create request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=(time_interval, time_interval)
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            bbox=bbox,
            size=size,
            config=config
        )
        
        # Get the data
        images = request.get_data()
        
        if not images or len(images) == 0:
            print(f"No image data found for date: {date.isoformat()}")
            return None
        
        # Check pixel values for debugging
        # print(f"Pixel values (sample): {images[0][0][0]}")  # Debugging line
        
        # Save the image
        image = Image.fromarray(images[0])
        
        # Add cloud cover information to filename if available
        cloud_info = ""
        if date_info.get("cloud_cover") is not None:
            cloud_info = f"_cloud{date_info['cloud_cover']:.1f}"
        
        image_path = os.path.join(output_path, f"satellite_image_{date.isoformat()}{cloud_info}.png")
        image.save(image_path)
        
        print(f"Image saved to: {image_path}")
        return image_path
    
    except Exception as e:
        print(f"Error fetching image for date {date.isoformat()}: {e}")
        return None

def generate_html_report(images, location_info, output_path):
    """Generate an HTML report with all retrieved images."""
    report_path = os.path.join(output_path, "satellite_imagery_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Historical Satellite Imagery Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .image-container {{ margin-bottom: 30px; }}
            .image-container img {{ max-width: 100%; border: 1px solid #ddd; }}
            h1, h2 {{ color: #333; }}
            .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Historical Satellite Imagery Report</h1>
        <div class="metadata">
            <p><strong>Location:</strong> {location_info}</p>
            <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Images ({len(images)})</h2>
    """
    
    # Sort images by date
    sorted_images = sorted(images, key=lambda x: x['date'])
    
    for img in sorted_images:
        image_filename = os.path.basename(img['path'])
        cloud_info = ""
        if img.get('cloud_cover') is not None:
            cloud_info = f"<br><strong>Cloud Cover:</strong> {img['cloud_cover']}%"
            
        html_content += f"""
        <div class="image-container">
            <h3>Date: {img['date'].strftime('%Y-%m-%d')}{cloud_info}</h3>
            <img src="{image_filename}" alt="Satellite image from {img['date'].strftime('%Y-%m-%d')}">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Fetch historical satellite imagery.')
    
    # Define location arguments (either address or coordinates)
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--address', type=str, help='Address to geocode')
    location_group.add_argument('--coordinates', type=str, help='Latitude,Longitude (comma-separated)')
    
    # Date range arguments
    parser.add_argument('--start-date', type=str, required=True, 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True, 
                        help='End date in YYYY-MM-DD format')
    
    # Optional arguments
    parser.add_argument('--area-size', type=float, default=2, 
                        help='Size of the area in km (default: 0.5)')
    parser.add_argument('--resolution', type=int, default=10, 
                        help='Resolution in meters per pixel (default: 10)')
    parser.add_argument('--max-cloud-cover', type=float, default=100, 
                        help='Maximum cloud cover percentage (default: 100)')
    parser.add_argument('--output-dir', type=str, default='./satellite_images', 
                        help='Output directory for images (default: ./satellite_images)')
    parser.add_argument('--data-source', type=str, default='SENTINEL2_L2A', 
                        choices=['SENTINEL2_L2A', 'SENTINEL2_L1C', 'LANDSAT8'], 
                        help='Data source (default: SENTINEL2_L2A)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to download (default: all available)')
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD format.")
        return 1
    
    # Get coordinates
    if args.address:
        print(f"Geocoding address: {args.address}")
        coordinates = address_to_coordinates(args.address)
        if not coordinates:
            print("Error: Could not geocode the address. Please provide coordinates directly.")
            return 1
        latitude, longitude = coordinates
        location_info = args.address
    else:
        try:
            latitude, longitude = map(float, args.coordinates.split(','))
            location_info = f"{latitude}, {longitude}"
        except ValueError:
            print("Error: Invalid coordinates format. Please provide as latitude,longitude")
            return 1
    
    print(f"Location: {latitude}, {longitude}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up Sentinel Hub configuration
    config = setup_sentinel_config()
    
    # Select the data collection
    data_collections = {
        'SENTINEL2_L2A': DataCollection.SENTINEL2_L2A,
        'SENTINEL2_L1C': DataCollection.SENTINEL2_L1C,
        'LANDSAT8': DataCollection.HARMONIZED_LANDSAT_SENTINEL
    }
    data_collection = data_collections[args.data_source]
    
    # Create bounding box
    bbox = create_bounding_box(latitude, longitude, args.area_size, args.area_size)
    
    # Get available dates
    print(f"Checking for available imagery between {start_date} and {end_date}...")
    available_dates = get_available_dates(config, bbox, data_collection, start_date, end_date)
    
    if not available_dates:
        print("No imagery available for the specified location and date range.")
        return 0
    
    # Filter by cloud cover if applicable
    if args.max_cloud_cover < 100:
        filtered_dates = [d for d in available_dates if d.get('cloud_cover') is None or d.get('cloud_cover') <= args.max_cloud_cover]
        print(f"Filtered from {len(available_dates)} to {len(filtered_dates)} images based on cloud cover <= {args.max_cloud_cover}%")
        available_dates = filtered_dates
    
    # Apply max images limit if specified
    if args.max_images is not None and len(available_dates) > args.max_images:
        print(f"Limiting to {args.max_images} images as requested")
        # Try to select evenly distributed dates
        if len(available_dates) > args.max_images > 1:
            step = len(available_dates) // args.max_images
            available_dates = available_dates[::step][:args.max_images]
        else:
            available_dates = available_dates[:args.max_images]
    
    # Fetch images
    images = []
    for date_info in available_dates:
        print(f"Fetching image for date: {date_info['date'].isoformat()}")
        image_path = fetch_sentinel_image(
            config, bbox, date_info, data_collection, output_path, args.resolution
        )
        
        if image_path:
            images.append({
                'date': date_info['date'],
                'path': image_path,
                'cloud_cover': date_info.get('cloud_cover')
            })
    
    if images:
        # Generate HTML report
        report_path = generate_html_report(images, location_info, output_path)
        print(f"\nSuccessfully retrieved {len(images)} images.")
        print(f"Report available at: {report_path}")
    else:
        print("No images were successfully retrieved.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())