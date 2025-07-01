# Traffic Analysis System

A Python-based system for simulating, analyzing, and visualizing urban traffic patterns, with a focus on Tehran, Iran. The project leverages real-world map data, clustering, and interactive mapping to identify traffic hot zones and suggest alternative routes.

## Features

- **Street Network Analysis:** Automatically fetches and processes the street network of Tehran (or any specified city).
- **Traffic & Demand Simulation:** Generates synthetic traffic and demand data for each node in the network.
- **Hot Zone Identification:** Uses KMeans clustering to classify areas into red (high), yellow (medium), and green (low) traffic/demand zones.
- **Interactive Visualization:** Creates an interactive HTML map with heatmaps and color-coded zone markers.
- **Alternative Route Finder:** Computes multiple alternative routes between high-traffic and low-traffic zones.
- **Extensible & Modular:** Easily adaptable to other cities or real traffic data.

## Demo

After running the system, an interactive map (`tehran_traffic_analysis.html`) will be generated, showing simulated traffic hot zones and a heatmap overlay.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd traffic-analysis-system
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply run the main script:

```bash
python traffic_analysis.py
```

- The script will:
  - Download the street network for Tehran.
  - Simulate traffic and demand data.
  - Identify and cluster hot zones.
  - Generate an interactive map: `tehran_traffic_analysis.html`.

You can open the generated HTML file in your browser to explore the results.

## Customization

- **Change City:**  
  Edit the `place` parameter in `TrafficManager`'s constructor to analyze a different city.
- **Adjust Clustering:**  
  Modify `NUM_CLUSTERS`, `TRAFFIC_WEIGHT`, or `DEMAND_WEIGHT` in `traffic_analysis.py` to tune the clustering behavior.
- **Real Data Integration:**  
  Replace the `_generate_simulation_data` method with real traffic/demand data sources for more accurate analysis.

## Project Structure

```
traffic_analysis.py         # Main analysis and visualization script
requirements.txt           # Python dependencies
tehran_traffic_analysis.html # Output map (generated after running the script)
README.md                  # Project documentation
```

## Dependencies

- folium==0.20.0
- pandas==2.3.0
- osmnx==2.0.4
- scikit-learn==1.7.0
- networkx==3.4.2

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is provided for educational and research purposes. Please check individual package licenses for third-party dependencies.

## Acknowledgments

- [OSMnx](https://github.com/gboeing/osmnx) for street network data.
- [Folium](https://python-visualization.github.io/folium/) for interactive mapping.
- [scikit-learn](https://scikit-learn.org/) for clustering.