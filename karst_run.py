"""
Command-line tool: selective depression filling and hydrological analysis of
a karst DEM. Writes the results as GeoTIFFs next to the input or into a
chosen folder.

Example:
    python karst_run.py -i dem.tif -o results
"""
import argparse
import os
import sys
import rasterio
from karst_dem_processor import KarstDEMProcessor, DEFAULT_THRESHOLDS


def write_raster(path, array, kdp, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=kdp.height, width=kdp.width,
                       count=1, dtype=array.dtype, crs=kdp.crs, transform=kdp.transform,
                       nodata=nodata) as dst:
        dst.write(array, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Selective depression filling and hydrological analysis for karst DEMs.")
    parser.add_argument('-i', '--input', required=True,
                        help="input DEM (GeoTIFF, projected CRS in metres)")
    parser.add_argument('-o', '--output_dir', default=None,
                        help="output folder (default: folder of the input DEM)")
    parser.add_argument('-d', '--depth', type=float, default=DEFAULT_THRESHOLDS['depth'],
                        help="minimum depth of a karst depression in m (default: %(default)s)")
    parser.add_argument('-a', '--area', type=float, default=DEFAULT_THRESHOLDS['area'],
                        help="minimum area in m^2 (default: %(default)s)")
    parser.add_argument('-v', '--volume', type=float, default=DEFAULT_THRESHOLDS['volume'],
                        help="minimum volume in m^3 (default: %(default)s)")
    parser.add_argument('-e', '--elongation', type=float, default=DEFAULT_THRESHOLDS['elongation'],
                        help="maximum elongation ratio (default: %(default)s)")
    parser.add_argument('-c', '--circularity', type=float, default=DEFAULT_THRESHOLDS['circularity'],
                        help="minimum circularity index for elongated depressions (default: %(default)s)")
    parser.add_argument('-t', '--threshold', type=float, default=500.0,
                        help="flow accumulation (cells) needed to start a stream (default: %(default)s)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(os.path.abspath(args.input))
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    thresholds = {
        'depth': args.depth,
        'area': args.area,
        'volume': args.volume,
        'elongation': args.elongation,
        'circularity': args.circularity,
    }

    print(f"Input: {args.input}")
    print(f"Thresholds: depth {args.depth} m, area {args.area} m^2, volume {args.volume} m^3, "
          f"elongation {args.elongation}, circularity {args.circularity}")

    print("[1/7] Loading DEM and detecting depressions")
    kdp = KarstDEMProcessor(args.input)
    kdp.detect_depressions()

    print("[2/7] Classifying depressions and selective filling (Burn-In)")
    kdp.classify_depressions(thresholds)
    export_dem = kdp.fill_dem()

    print("[3/7] D8 flow direction")
    flow_dir = kdp.calculate_flow_direction()

    print("[4/7] Flow accumulation")
    flow_acc = kdp.calculate_flow_accumulation()

    print("[5/7] Stream network")
    streams = kdp.extract_streams(threshold=args.threshold)

    print("[6/7] Strahler order")
    strahler = kdp.calculate_strahler_order(threshold=args.threshold)

    print("[7/7] Watershed basins")
    basins = kdp.delineate_basins()

    outputs = [
        ("export_dem", export_dem, kdp.nodata),
        ("routing_dem", kdp.routing_dem.astype(export_dem.dtype), kdp.nodata),
        ("flowdir", flow_dir, 0),
        ("flowacc", flow_acc, -1),
        ("streams", streams, 0),
        ("strahler", strahler, 0),
        ("basins", basins, 0),
    ]
    for name, array, nodata in outputs:
        write_raster(os.path.join(out_dir, f"{base_name}_{name}.tif"), array, kdp, nodata)

    n_total = len(kdp.depressions)
    n_karst = sum(1 for dep in kdp.depressions if dep.is_karst)
    print(f"\nDepressions: {n_total} (karst: {n_karst}, artefacts: {n_total - n_karst})")
    print(f"Outputs written to {out_dir}:")
    for name, _, _ in outputs:
        print(f"  {base_name}_{name}.tif")


if __name__ == "__main__":
    main()
