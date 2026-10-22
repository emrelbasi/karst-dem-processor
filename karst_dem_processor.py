"""
Selective depression filling and D8 flow routing for karst DEMs.

Depressions found by morphological reconstruction are classified as karst
features or artefacts from their depth, area, volume, elongation and
circularity. Artefacts are filled, while each karst depression is kept as a
closed sink.

Reference:
    Elbaşı, E. (2026). Classification-driven selective depression filling for
    hydrological DEM conditioning in karst terrains. Environmental Modelling
    & Software.
"""
import numpy as np
import rasterio
import logging
import warnings
from skimage import measure
from typing import Optional, List, Dict, Any
from numba import njit, prange

# D8 code -> (row, col) offset, indexed directly by the code (1, 2, 4, ..., 128).
_D8_DR = np.zeros(129, dtype=np.int8)
_D8_DC = np.zeros(129, dtype=np.int8)
_D8_VALID = np.zeros(129, dtype=np.bool_)
_d8_map = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
           16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
for _code, (_dr, _dc) in _d8_map.items():
    _D8_DR[_code] = _dr
    _D8_DC[_code] = _dc
    _D8_VALID[_code] = True

# Default classification thresholds: depth (m), area (m²), volume (m³),
# maximum elongation ratio and minimum circularity index.
DEFAULT_THRESHOLDS = {
    'depth': 2.0,
    'area': 90000.0,
    'volume': 45000.0,
    'elongation': 3.0,
    'circularity': 0.4,
}

logger = logging.getLogger('KarstDEMProcessor')


class KarstProcessingError(Exception):
    """Base class for processing errors."""
    pass

class DEMValidationError(KarstProcessingError):
    """The input DEM cannot be processed."""
    pass

class TopologicalError(KarstProcessingError):
    """Inconsistent flow topology (unresolved flats, cycles)."""
    pass


class Depression:
    """
    A closed depression and the properties used to classify it.

    Attributes:
        coords: Centre of the bounding box (row, col).
        depth: Maximum depth below the spill level (m).
        area_px: Number of cells.
        area_m2: Area (m²).
        volume_m3: Volume below the spill level (m³).
        is_karst, is_artifact: Classification result.
        elongation: Major axis / minor axis of the fitted ellipse.
        circularity: 4*pi*area / perimeter^2.
        label_id: Connected-component label of the depression.
        sink_row, sink_col: Lowest cell of the depression.
    """
    def __init__(self, coords: tuple, depth: float, area_px: int,
                 area_m2: float, volume_m3: float,
                 is_karst: bool = False, is_artifact: bool = True,
                 elongation: float = 0.0, circularity: float = 0.0,
                 label_id: int = 0, sink_row: int = -1, sink_col: int = -1):
        self.coords = coords
        self.depth = depth
        self.area_px = area_px
        self.area_m2 = area_m2
        self.volume_m3 = volume_m3
        self.is_karst = is_karst
        self.is_artifact = is_artifact
        self.elongation = elongation
        self.circularity = circularity
        self.label_id = label_id
        self.sink_row = sink_row
        self.sink_col = sink_col

    def __repr__(self):
        return (f"Depression(label={self.label_id}, coords={self.coords}, "
                f"depth={self.depth:.2f}, area_px={self.area_px}, "
                f"area_m2={self.area_m2:.1f}, volume_m3={self.volume_m3:.2f}, "
                f"is_karst={self.is_karst}, is_artifact={self.is_artifact}, "
                f"elongation={self.elongation:.2f}, circularity={self.circularity:.2f})")


@njit(parallel=True, cache=True)
def _calculate_d8_numba(dem, karst_mask, nodata_mask):
    """D8 steepest-descent flow direction, parallel over rows."""
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    offsets_r = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    offsets_c = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    codes = np.array([64, 128, 1, 2, 4, 8, 16, 32], dtype=np.uint8)

    for r in prange(1, rows - 1):
        for c in range(1, cols - 1):
            if karst_mask[r, c] or nodata_mask[r, c]:
                continue

            current_elev = dem[r, c]
            max_slope = -np.inf
            best_dir = np.uint8(0)

            is_boundary = (r <= 1 or r >= rows - 2 or c <= 1 or c >= cols - 2)

            for i in range(8):
                dr = offsets_r[i]
                dc = offsets_c[i]
                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                dist = 1.41421356 if (abs(dr) == 1 and abs(dc) == 1) else 1.0

                if nodata_mask[nr, nc]:
                    # Cells next to the grid edge may drain into NoData (sea or
                    # outside the study area); interior cells may not, so that
                    # gaps in the data do not act as outlets.
                    if is_boundary:
                        slope = 1e9 / dist
                    else:
                        continue
                else:
                    slope = (current_elev - dem[nr, nc]) / dist

                if slope > max_slope:
                    max_slope = slope
                    best_dir = codes[i]

            if max_slope > 0:
                flow_dir[r, c] = best_dir
            else:
                flow_dir[r, c] = 0

    return flow_dir


@njit(parallel=False, cache=True)
def _resolve_flats_numba(dem, flow_dir, karst_mask):
    """
    Assign flow directions on flats by breadth-first search that starts at
    the cells through which each flat drains and proceeds inward.
    """
    rows, cols = dem.shape
    offsets_r = [-1, -1, 0, 1, 1, 1, 0, -1]
    offsets_c = [0, 1, 1, 1, 0, -1, -1, -1]
    opposite_codes = [4, 8, 16, 32, 64, 128, 1, 2]

    max_elements = rows * cols
    queue = np.zeros(max_elements, dtype=np.int32)
    head = 0
    tail = 0

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if flow_dir[r, c] != 0 and not karst_mask[r, c]:
                for i in range(8):
                    nr = r + offsets_r[i]
                    nc = c + offsets_c[i]

                    if nr < 1 or nr >= rows - 1 or nc < 1 or nc >= cols - 1:
                        continue

                    if flow_dir[nr, nc] == 0 and not karst_mask[nr, nc]:
                        if dem[nr, nc] == dem[r, c]:
                            queue[tail] = r * cols + c
                            tail += 1
                            break

    while head < tail:
        idx = queue[head]
        head += 1

        r = idx // cols
        c = idx % cols

        for i in range(8):
            nr = r + offsets_r[i]
            nc = c + offsets_c[i]

            if nr < 1 or nr >= rows - 1 or nc < 1 or nc >= cols - 1:
                continue

            if flow_dir[nr, nc] == 0 and not karst_mask[nr, nc]:
                if dem[nr, nc] == dem[r, c]:
                    flow_dir[nr, nc] = opposite_codes[i]
                    queue[tail] = nr * cols + nc
                    tail += 1

    return flow_dir


@njit(parallel=False, cache=True)
def _calculate_accumulation_numba(flow_dir, nodata_mask, d8_dr, d8_dc, d8_valid):
    """
    Flow accumulation (in cells) in topological order (Kahn's algorithm),
    which avoids recursion on large grids.
    """
    rows, cols = flow_dir.shape

    accumulation = np.ones((rows, cols), dtype=np.float32)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue

            dir_val = flow_dir[r, c]
            if dir_val == 0 or dir_val > 128 or not d8_valid[dir_val]:
                continue

            nr = r + d8_dr[dir_val]
            nc = c + d8_dc[dir_val]

            if 0 <= nr < rows and 0 <= nc < cols:
                in_degree[nr, nc] += 1

    max_elements = rows * cols
    queue = np.zeros(max_elements, dtype=np.int32)
    head = 0
    tail = 0

    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue
            if in_degree[r, c] == 0:
                queue[tail] = r * cols + c
                tail += 1

    while head < tail:
        idx = queue[head]
        head += 1

        r = idx // cols
        c = idx % cols

        dir_val = flow_dir[r, c]

        if dir_val == 0 or dir_val > 128 or not d8_valid[dir_val]:
            continue

        nr = r + d8_dr[dir_val]
        nc = c + d8_dc[dir_val]

        if 0 <= nr < rows and 0 <= nc < cols and not nodata_mask[nr, nc]:
            accumulation[nr, nc] += accumulation[r, c]

            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                queue[tail] = nr * cols + nc
                tail += 1

    return accumulation


@njit(parallel=False, cache=True)
def _calculate_strahler_numba(flow_dir, accumulation, threshold, d8_dr, d8_dc, d8_valid):
    """Strahler order of stream cells (accumulation >= threshold), in topological order."""
    rows, cols = flow_dir.shape
    strahler = np.zeros((rows, cols), dtype=np.int32)
    in_degree = np.zeros((rows, cols), dtype=np.int32)
    max_order_in = np.zeros((rows, cols), dtype=np.int32)
    count_max_order = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            dir_val = flow_dir[r, c]
            if dir_val == 0 or dir_val > 128 or not d8_valid[dir_val]:
                continue

            nr = r + d8_dr[dir_val]
            nc = c + d8_dc[dir_val]

            if 0 <= nr < rows and 0 <= nc < cols:
                in_degree[nr, nc] += 1

    queue = np.zeros(rows * cols, dtype=np.int32)
    head = 0
    tail = 0

    for r in range(rows):
        for c in range(cols):
            if in_degree[r, c] == 0:
                queue[tail] = r * cols + c
                tail += 1

    while head < tail:
        idx = queue[head]
        head += 1
        r = idx // cols
        c = idx % cols

        if accumulation[r, c] < threshold:
            current_order = 0
        else:
            m_in = max_order_in[r, c]
            c_in = count_max_order[r, c]
            if m_in == 0:
                current_order = 1
            elif c_in >= 2:
                current_order = m_in + 1
            else:
                current_order = m_in

        strahler[r, c] = current_order

        dir_val = flow_dir[r, c]
        if dir_val == 0 or dir_val > 128 or not d8_valid[dir_val]:
            continue

        nr = r + d8_dr[dir_val]
        nc = c + d8_dc[dir_val]

        if 0 <= nr < rows and 0 <= nc < cols:
            if current_order > 0:
                if current_order > max_order_in[nr, nc]:
                    max_order_in[nr, nc] = current_order
                    count_max_order[nr, nc] = 1
                elif current_order == max_order_in[nr, nc]:
                    count_max_order[nr, nc] += 1

            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                queue[tail] = nr * cols + nc
                tail += 1

    return strahler


@njit(parallel=False, cache=True)
def _delineate_basins_numba(flow_dir, nodata_mask):
    """
    Label the area draining to each terminal cell (flow direction 0) by
    reverse BFS. NoData cells never start a basin.
    """
    rows, cols = flow_dir.shape
    basins = np.zeros((rows, cols), dtype=np.int32)
    queue = np.zeros(rows * cols, dtype=np.int32)
    head = 0
    tail = 0

    basin_id = 1
    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue
            if flow_dir[r, c] == 0:
                basins[r, c] = basin_id
                queue[tail] = r * cols + c
                tail += 1
                basin_id += 1

    offsets_r = [-1, -1, 0, 1, 1, 1, 0, -1]
    offsets_c = [0, 1, 1, 1, 0, -1, -1, -1]
    req_dirs = [4, 8, 16, 32, 64, 128, 1, 2]

    while head < tail:
        idx = queue[head]
        head += 1
        r = idx // cols
        c = idx % cols
        current_basin = basins[r, c]

        for i in range(8):
            nr = r + offsets_r[i]
            nc = c + offsets_c[i]
            req_dir = req_dirs[i]

            if 0 <= nr < rows and 0 <= nc < cols:
                if nodata_mask[nr, nc]:
                    continue
                if basins[nr, nc] == 0 and flow_dir[nr, nc] == req_dir:
                    basins[nr, nc] = current_basin
                    queue[tail] = nr * cols + nc
                    tail += 1

    return basins


class KarstDEMProcessor:
    """
    Selective depression filling and hydrological analysis of a karst DEM.

    Typical use:
        kdp = KarstDEMProcessor('dem.tif')
        kdp.detect_depressions()
        kdp.classify_depressions()
        export_dem = kdp.fill_dem()      # kdp.routing_dem is set as well
        flow_dir = kdp.calculate_flow_direction()
        flow_acc = kdp.calculate_flow_accumulation()

    The DEM should be in a projected CRS with metric units, because the
    area and volume thresholds are given in m² and m³.
    """

    def __init__(self, dem_path: Optional[str] = None):
        self.dem_data: Optional[np.ndarray] = None
        self.nodata: Optional[float] = None
        self.transform: Optional[rasterio.Affine] = None
        self.crs: Optional[rasterio.CRS] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.filled_dem: Optional[np.ndarray] = None
        self.routing_dem: Optional[np.ndarray] = None
        self.difference_dem: Optional[np.ndarray] = None
        self.flow_dir: Optional[np.ndarray] = None
        self.depressions: List[Depression] = []
        self.pixel_area_m2: Optional[float] = None
        self._labeled_depressions: Optional[np.ndarray] = None
        self._nodata_mask: Optional[np.ndarray] = None

        if dem_path:
            self.load_dem(dem_path)

    @property
    def labeled_depressions(self) -> np.ndarray:
        """
        Connected-component labels of the depressions. Computed once, so that
        detection, filling and routing all refer to the same labels.
        """
        if self._labeled_depressions is None:
            if self.difference_dem is None:
                self.difference_dem = self._calculate_difference_dem()
            depression_mask = self.difference_dem > 0
            self._labeled_depressions, _ = measure.label(
                depression_mask, connectivity=2, return_num=True
            )
        return self._labeled_depressions

    @property
    def nodata_mask(self) -> np.ndarray:
        """Boolean mask of NoData (and NaN) cells."""
        if self._nodata_mask is None:
            if self.dem_data is None:
                raise ValueError("No DEM data loaded")
            if self.nodata is not None:
                self._nodata_mask = (self.dem_data == self.nodata) | np.isnan(self.dem_data)
            else:
                self._nodata_mask = np.isnan(self.dem_data)
        return self._nodata_mask

    def load_dem(self, dem_path: str) -> None:
        """
        Read the first band of a raster and check that it can be processed.

        Raises:
            DEMValidationError: if the DEM is entirely NoData or has a single value.
        """
        with rasterio.open(dem_path) as src:
            self.dem_data = src.read(1)
            self.nodata = src.nodata
            self.transform = src.transform
            self.crs = src.crs
            self.width = src.width
            self.height = src.height

        self._labeled_depressions = None
        self._nodata_mask = None
        self.difference_dem = None
        self.filled_dem = None
        self.routing_dem = None
        self.flow_dir = None
        self.depressions = []

        # Cell area from the affine determinant, which also holds for rotated grids.
        t = self.transform
        if t is not None:
            self.pixel_area_m2 = abs(t.a * t.e - t.b * t.d)
        else:
            self.pixel_area_m2 = 1.0

        if self.crs is not None and self.crs.is_geographic:
            warnings.warn(
                "The DEM has a geographic CRS (degrees), so depression areas and "
                "volumes are not in m² and m³. Reproject it to a metric CRS "
                "(e.g. UTM) first.",
                UserWarning
            )

        valid_mask = ~self.nodata_mask
        valid_count = int(np.sum(valid_mask))

        if valid_count == 0:
            raise DEMValidationError(
                "The DEM contains only NoData cells."
            )

        unique_vals = np.unique(self.dem_data[valid_mask])
        if len(unique_vals) <= 1:
            raise DEMValidationError(
                f"The DEM is completely flat (single value: {unique_vals[0]:.2f}); "
                "depressions and flow directions cannot be derived."
            )

        elevation_range = float(np.ptp(self.dem_data[valid_mask]))
        if elevation_range < 0.01:
            warnings.warn(
                f"The elevation range of the DEM is very small ({elevation_range:.4f} m); "
                "results may not be reliable.",
                UserWarning
            )

        logger.info(
            f"DEM loaded: {self.height}x{self.width}, "
            f"valid cells: {valid_count}, "
            f"elevation range: {elevation_range:.2f} m"
        )

    def _fill_dem(self) -> np.ndarray:
        """Standard filling of all depressions (reconstruction by erosion)."""
        if self.dem_data is None:
            raise ValueError("No DEM data loaded")

        from skimage.morphology import reconstruction
        dem = self.dem_data.astype(np.float32)

        seed = np.copy(dem)
        max_val = np.nanmax(dem)

        seed[1:-1, 1:-1] = max_val

        if self.nodata is not None:
            seed = np.where(dem == self.nodata, dem, seed)
            dem_mask = dem
        else:
            dem_mask = dem

        seed = np.maximum(seed, dem_mask)

        filled_dem = reconstruction(seed, dem_mask, method='erosion')

        if self.nodata is not None:
            filled_dem[dem == self.nodata] = self.nodata

        return filled_dem.astype(np.float32)

    def _calculate_difference_dem(self) -> np.ndarray:
        """Depth of filling: standard-filled DEM minus original DEM."""
        if self.dem_data is None:
            raise ValueError("No DEM data loaded")
        if self.filled_dem is None:
            self.filled_dem = self._fill_dem()
        difference_dem = self.filled_dem.astype(np.float32) - self.dem_data.astype(np.float32)
        return difference_dem

    def detect_depressions(self) -> List[Depression]:
        """
        Find all depressions (cells raised by standard filling) and measure
        their depth, area, volume and shape.
        """
        if self.dem_data is None:
            raise ValueError("No DEM data loaded")
        self.difference_dem = self._calculate_difference_dem()

        labeled_depressions = self.labeled_depressions

        regions = measure.regionprops(labeled_depressions, intensity_image=self.difference_dem)

        depressions = []
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            center_row = (min_row + max_row) // 2
            center_col = (min_col + max_col) // 2
            coords = (center_row, center_col)

            depression_values = region.image_intensity[region.image]

            max_depth = np.max(depression_values)
            area_px = int(region.area)
            pixel_area = self.pixel_area_m2 if self.pixel_area_m2 is not None else 1.0
            area_m2 = area_px * pixel_area
            volume_m3 = float(np.sum(depression_values) * pixel_area)

            major_axis = region.axis_major_length
            minor_axis = region.axis_minor_length
            elongation = major_axis / (minor_axis + 1e-6)

            perimeter = region.perimeter
            circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0.0

            # The sink is the lowest cell of the depression; on ties, the first
            # one in row-major order.
            sub_dem = self.dem_data[min_row:max_row, min_col:max_col]
            masked_dem = np.where(region.image, sub_dem, np.inf)
            _flat = int(np.argmin(masked_dem))
            _lr, _lc = np.unravel_index(_flat, masked_dem.shape)
            sink_row = min_row + int(_lr)
            sink_col = min_col + int(_lc)

            dep = Depression(
                coords, max_depth, area_px, area_m2, volume_m3,
                is_karst=False, is_artifact=True,
                elongation=elongation, circularity=circularity,
                label_id=region.label,
                sink_row=sink_row, sink_col=sink_col
            )
            depressions.append(dep)

        self.depressions = depressions
        return depressions

    def classify_depressions(self, karst_thresholds: Optional[Dict[str, float]] = None) -> List[Depression]:
        """
        Classify each depression as karst or artefact.

        A depression is karst when its depth, area and volume reach the
        thresholds and it is either compact (elongation <= 'elongation') or
        has a regular outline (circularity > 'circularity'). Keys missing
        from ``karst_thresholds`` take the values in DEFAULT_THRESHOLDS.
        """
        if not self.depressions:
            self.detect_depressions()

        t = dict(DEFAULT_THRESHOLDS)
        if karst_thresholds:
            t.update(karst_thresholds)

        for dep in self.depressions:
            if (dep.depth >= t['depth'] and
                dep.area_m2 >= t['area'] and
                dep.volume_m3 >= t['volume'] and
                (dep.elongation <= t['elongation'] or dep.circularity > t['circularity'])):
                dep.is_karst = True
                dep.is_artifact = False
            else:
                dep.is_karst = False
                dep.is_artifact = True

        return self.depressions

    def fill_dem(self) -> np.ndarray:
        """
        Selective filling: artefacts are filled, karst depressions are kept.

        Burn-In lowers the sink of every karst depression in both the marker
        and the mask of a second reconstruction, so that each karst depression
        gets an internal outlet. Artefacts are then filled exactly as in
        standard filling, while inside a karst depression the terrain is left
        unchanged except for internal sub-depressions, which are raised only
        to the level at which they spill toward the sink. This surface is
        stored as ``self.routing_dem`` and used for all flow routing.

        Returns the export DEM: the routing DEM with the original elevations
        restored over every karst depression.
        """
        if self.dem_data is None:
            raise ValueError("No DEM data loaded")
        if not self.depressions:
            self.detect_depressions()
            self.classify_depressions()

        filled_dem = self.dem_data.copy()

        if self.difference_dem is not None and self.depressions:
            seed = np.copy(self.dem_data)
            seed[1:-1, 1:-1] = np.max(self.dem_data)

            mask = np.copy(self.dem_data)

            for dep in self.depressions:
                if dep.is_karst and dep.sink_row >= 0:
                    min_row, min_col = dep.sink_row, dep.sink_col

                    # Any positive burn depth gives the same routing DEM;
                    # this one simply keeps a wide margin.
                    burn_depth = max(100.0, dep.depth * 2.0 + 50.0)

                    seed[min_row, min_col] = self.dem_data[min_row, min_col] - burn_depth
                    mask[min_row, min_col] = self.dem_data[min_row, min_col] - burn_depth

            from skimage.morphology import reconstruction
            self.routing_dem = reconstruction(seed, mask, method='erosion')
            filled_dem = self.routing_dem.copy()

            # Restore the whole karst depression, not only its sink: internal
            # sub-depressions are raised in the routing DEM but should keep
            # their true elevations in the export DEM.
            karst_ids = [dep.label_id for dep in self.depressions if dep.is_karst]
            if karst_ids:
                labels = self.labeled_depressions
                lut = np.zeros(int(labels.max()) + 1, dtype=bool)
                lut[karst_ids] = True
                karst_region = lut[labels]
                filled_dem[karst_region] = self.dem_data[karst_region]

        else:
            filled_dem = self.filled_dem.copy()
            self.routing_dem = filled_dem.copy()

        self.filled_dem = filled_dem
        return filled_dem

    def get_depression_stats(self) -> Dict[str, Any]:
        """Counts and totals of the classified depressions (areas in m², volumes in m³)."""
        if not self.depressions:
            return {"total_depressions": 0, "karst_count": 0, "artifact_count": 0}

        karst_count = sum(1 for dep in self.depressions if dep.is_karst)
        artifact_count = len(self.depressions) - karst_count
        total_volume = sum(dep.volume_m3 for dep in self.depressions)
        total_area = sum(dep.area_m2 for dep in self.depressions)
        avg_depth = float(np.mean([dep.depth for dep in self.depressions])) if self.depressions else 0.0

        return {
            "total_depressions": len(self.depressions),
            "karst_count": karst_count,
            "artifact_count": artifact_count,
            "total_volume": total_volume,
            "total_area": total_area,
            "average_depth": avg_depth
        }

    def calculate_flow_direction(self) -> np.ndarray:
        """
        D8 flow direction on the routing DEM. Karst sinks are given direction
        0 so that they act as terminal cells; flats are then resolved.
        """
        if self.filled_dem is None:
            self.fill_dem()

        karst_mask = np.zeros_like(self.dem_data, dtype=bool)
        if self.difference_dem is not None and self.depressions:
            for dep in self.depressions:
                if dep.is_karst and dep.sink_row >= 0:
                    karst_mask[dep.sink_row, dep.sink_col] = True

        dem_float64 = self.routing_dem.astype(np.float64)

        nodata_mask_f64 = self.nodata_mask.copy()

        flow_dir = _calculate_d8_numba(dem_float64, karst_mask, nodata_mask_f64)
        flow_dir = _resolve_flats_numba(dem_float64, flow_dir, karst_mask)

        self.flow_dir = flow_dir
        return flow_dir

    def calculate_flow_accumulation(self) -> np.ndarray:
        """Flow accumulation in number of cells, including the cell itself."""
        if self.flow_dir is None:
            self.calculate_flow_direction()

        nodata_mask = self.nodata_mask

        flow_acc = _calculate_accumulation_numba(
            self.flow_dir, nodata_mask, _D8_DR, _D8_DC, _D8_VALID
        )

        self.flow_acc = flow_acc
        return flow_acc

    def extract_streams(self, threshold: float = 500.0) -> np.ndarray:
        """Stream cells (1) where flow accumulation reaches ``threshold`` cells."""
        if getattr(self, 'flow_acc', None) is None:
            self.calculate_flow_accumulation()

        streams = np.zeros_like(self.flow_acc, dtype=np.uint8)
        streams[self.flow_acc >= threshold] = 1
        return streams

    def calculate_strahler_order(self, threshold: float = 500.0) -> np.ndarray:
        """Strahler order of the stream network defined by ``threshold``."""
        if getattr(self, 'flow_acc', None) is None:
            self.calculate_flow_accumulation()

        return _calculate_strahler_numba(
            self.flow_dir, self.flow_acc, threshold, _D8_DR, _D8_DC, _D8_VALID
        )

    def delineate_basins(self) -> np.ndarray:
        """Basin label for every cell, one basin per terminal cell."""
        if self.flow_dir is None:
            self.calculate_flow_direction()

        return _delineate_basins_numba(self.flow_dir, self.nodata_mask)
