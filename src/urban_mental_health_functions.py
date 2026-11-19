import os
import json
import rasterio
#from rasterstats import zonal_stats
from rasterio.enums import Resampling
from rasterio.windows import Window
import rasterio.features
import rasterio.windows
import rasterio.warp
import numpy as np
from tqdm import tqdm
#import gc
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr
import pyproj
import shapely
from shapely.geometry import MultiPolygon
#import pygeoprocessing as pgp
from hazelbean.spatial_utils import warp_raster_to_match
import rioxarray
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster, Lock
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock


def warp_raster_to_reference(
    input_path,
    reference_path,
    out_path,
    resampling_method='near',
    compress="lzw"
    ):
    """
    Reproject and resample a raster to match the resolution and grid of a reference raster.
    """

    if os.path.exists(out_path):
        print(f"Raster already exists at {out_path}. Skipping reprojection and resampling.")
        return out_path

    # result = warp_raster_to_match(
    #     input_path=input_path,
    #     output_path=out_path,
    #     match_path=reference_path,
    #     resample_method=resampling_method
    # )  # reconstructs output width & height from bounding_box/pixel_size

    # Forcing the output width,height to exactly equal the reference raster's
    # own RasterXSize/RasterYSize (read directly from its header) rather than
    # reconstruction

    with rasterio.open(reference_path) as ref:
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_crs_wkt = ref.crs.to_wkt()

    with rasterio.open(input_path) as src:
        src_nodata = src.nodata

    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS=ref_crs_wkt,
        outputBounds=(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top),
        width=ref_width,
        height=ref_height,
        resampleAlg=resampling_method,
        srcNodata=src_nodata,
        dstNodata=src_nodata,
        creationOptions=['TILED=YES', 'BIGTIFF=YES', f'COMPRESS={compress.upper()}'],
    )

    result = gdal.Warp(out_path, input_path, options=warp_options)
    result.FlushCache()
    result = None

    print(f"Reprojected and resampled raster saved to: {out_path}")
    return out_path


# This version of map_lulc_to_ndvi() is a disguised for loop: for each pixel, lc_to_ndvi.get() is called.
# This is millions of slow python function calls.
# def map_lulc_to_ndvi(lulc_path, attr_table_path, ndvi_col, out_path, compress="lzw"):
#     """
#     Map LULC raster codes to NDVI values using attribute table.
#     """

#     # Read attribute table
#     df = pd.read_csv(attr_table_path)

#     # Create lookup dictionary with NaN handling for unmapped codes
#     lc_to_ndvi = dict(zip(df['lc_code'], df[ndvi_col]))

#     with rasterio.open(lulc_path) as src:
#         profile = src.profile.copy()
#         profile.update(dtype='float32', BIGTIFF='YES', nodata=np.nan, compress=compress)

#         with rasterio.open(out_path, 'w', **profile) as dst:
#             windows = list(src.block_windows(1))
#             for idx, window in tqdm(windows, desc="Converting LULC to NDVI"):
#             #for idx, window in windows:
#                 arr = src.read(1, window=window)

#                 # Map LULC codes to NDVI values, handling unmapped codes as NaN
#                 ndvi_arr = np.vectorize(lambda x: lc_to_ndvi.get(x, np.nan), otypes=[float])(arr)  # np.vectorize for better performance with large arrays thus avoiding the need for a loop over each pixel

#                 dst.write(ndvi_arr.astype('float32'), 1, window=window)

#     #print(f"NDVI raster saved to: {out_path}")

# This version of map_lulc_to_ndvi() is better than the one above because pgp library
# is fundamentally a set of Python wrappers around GDAL's core C++ functions.
# def map_lulc_to_ndvi(lulc_path, attr_table_path, ndvi_col, out_path, compress="lzw"):
#     """
#     Map LULC raster codes to NDVI values using attribute table.
#     """

#     # Read attribute table
#     df = pd.read_csv(attr_table_path)

#     # Create lookup dictionary with NaN handling for unmapped codes
#     lc_to_ndvi = dict(zip(df['lc_code'], df[ndvi_col]))

#     # Pre-scan the LULC raster to find ALL unique values
#     unique_raster_values = set()
#     with rasterio.open(lulc_path) as src:
#         src_nodata = src.nodata
#         for idx, window in src.block_windows(1):
#             block = src.read(1, window=window)
#             unique_raster_values.update(np.unique(block))

#     # Add any value from the raster that is not in our table
#     # to map these unmapped codes to np.nan
#     # in order to complete the dictionary
#     for val in unique_raster_values:
#         if val not in lc_to_ndvi:
#             lc_to_ndvi[int(val)] = np.nan # Add the missing key

#     # Run the optimized reclassification
#     pgp.reclassify_raster(
#         (lulc_path, 1),      # (path, band_index) for base raster
#         lc_to_ndvi,          # The lookup dictionary
#         out_path,            # Output path
#         gdal.GDT_Float32,    # Output data type
#         np.nan               # Output nodata value
#     )

#     print(f"NDVI raster saved to: {out_path}")

# This version of map_lulc_to_ndvi() has multhreaded implementation.
def map_lulc_to_ndvi_dask(lulc_path, attr_table_path, ndvi_col, out_path, compress='deflate'):
    """
    Map LULC raster codes to NDVI values using attribute table.
    """

    if os.path.exists(out_path):
        print(f"Raster already exists at {out_path}. Skipping LULC to NDVI conversion.")
        return out_path

    with rasterio.open(lulc_path) as src:
        src_nodata = src.nodata

    # Setup Dask cluster.
    with LocalCluster(
        #n_workers=os.cpu_count()//4 or 1,
        n_workers=8,
        threads_per_worker=1,
        memory_limit='45GB',
        death_timeout='2400s'
        ) as cluster, \
         Client(cluster) as client:
        
        print(f"Dask Dashboard link: {client.dashboard_link}")

        # Create the lookup array.
        df = pd.read_csv(attr_table_path)
        lc_codes = df['lc_code'].values.astype(int)
        ndvi_values = df[ndvi_col].values.astype(np.float32)
        max_code = np.max(lc_codes)
        lookup_array = np.full(int(max_code) + 1, np.nan, dtype=np.float32)
        lookup_array[lc_codes] = ndvi_values
        
        # Define the function to apply to each block.
        def lookup(block):
            max_lookup_index = len(lookup_array) - 1
            
            ndvi_arr = np.full(block.shape, np.nan, dtype=np.float32)  # fast, C-optimized NumPy lookup
            
            if block.size > 0:
                mask = (block <= max_lookup_index) & (block >= 0)
                if src_nodata is not None:
                    mask = mask & (block != src_nodata)  # exclude the nodata value
                valid_codes = block[mask]
                ndvi_arr[mask] = lookup_array[valid_codes]
            
            return ndvi_arr

        src_raster = rioxarray.open_rasterio(
            lulc_path, 
            chunks={'x': 2048, 'y': 2048},  # tell Dask how to chunk
            lock=False
        )  # open_rasterio opens the GeoTIFF
            # and returns a Dask-backed XARRAY DataArray
        
        # Apply lookup() to src_raster.
        ndvi_result = xr.apply_ufunc(
            lookup,
            src_raster,
            dask="parallelized",
            output_dtypes=[np.float32]
        )
        
        # Copy the spatial metadata from the source to the result.
        ndvi_result = ndvi_result.rio.write_crs(src_raster.rio.crs)
        ndvi_result = ndvi_result.rio.write_transform(src_raster.rio.transform())
        ndvi_result.rio.update_encoding(dict(dtype='float32', _FillValue=np.nan, nodata=np.nan, compress=compress, BIGTIFF='YES'), inplace=True)

        # Execute the job and save to disk.
        print("Starting parallel LULC to NDVI conversion...")
        with rasterio.Env(
            CHECK_DISK_FREE_SPACE=False,
            GDAL_CACHEMAX=512,
            GDAL_NUM_THREADS=1
            ):
            with ProgressBar():
                ndvi_result.rio.to_raster(
                    out_path,
                    tiled=True,
                    #lock=dask.utils.SerializableLock(),
                    lock=Lock("rio-lock", client=client),
                    compress=compress,
                    compute=True
                    )
        src_raster.close()
        print(f"NDVI raster saved to: {out_path}")


def create_no_vegetation_ndvi_scenario(
        baseline_ndvi_path,
        out_path,
        compress="deflate"
        ):
    """
    Create a no-vegetation scenario by setting all positive NDVI values to 0.
    Keeps built-up areas, water, etc. at their original NDVI values.
    """
    
    if os.path.exists(out_path):
        print(f"Raster already exists at {out_path}. Skipping creation of no vegetation NDVI scenario.")
        return out_path
    
    with rasterio.open(baseline_ndvi_path) as src:
        profile = src.profile.copy()
        profile.update(compress=compress, dtype='float32', BIGTIFF='YES')
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            for _, window in tqdm(src.block_windows(1), desc="Creating no-vegetation NDVI scenario"):
                ndvi_data = src.read(1, window=window).astype('float32')
                
                # Set positive NDVI (vegetation) to 0, keep negative/zero values
                no_veg_data = np.where(ndvi_data > 0.0, 0.0, ndvi_data)
                
                dst.write(no_veg_data, 1, window=window)
    
    print(f"No-vegetation NDVI scenario saved to: {out_path}")


def calculate_delta_raster(
        raster1_path,
        raster2_path,
        out_path,
        operation,
        fill_value=np.nan,
        #compress="lzw"
        compress="deflate"
        ):
    """
    Calculate difference between two rasters.
    """

    if os.path.exists(out_path):
        print(f"Raster already exists at {out_path}. Skipping delta raster computation.")
        return out_path
    
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        if src1.shape != src2.shape:
            raise ValueError("Input rasters must have the same dimensions.")

        profile = src1.profile.copy()
        profile.update(dtype='float32', BIGTIFF='YES', nodata=fill_value, compress=compress)

        with rasterio.open(out_path, "w", **profile) as dst:
            windows = list(src1.block_windows(1))
            #for _, window in windows:
            for _, window in tqdm(windows, desc="Calculating delta raster"):
                arr1 = src1.read(1, window=window)
                arr2 = src2.read(1, window=window)

                # Handle NaN values properly
                result = operation(arr1, arr2)
                result = np.where(np.isnan(arr1) | np.isnan(arr2), fill_value, result)

                dst.write(result.astype("float32"), 1, window=window)

    #print(f"Delta raster saved to: {out_path}")


def calculate_preventable_cases(
        delta_ne_path,
        pop_path,
        effect_size_table_path,
        prevalence,
        p0,
        out_path,
        #compress="lzw"
        compress="deflate"
        ):
    """
    Calculate preventable cases per pixel using delta nature exposure and population.
    """

    if os.path.exists(out_path):
        print(f"Raster already exists at {out_path}. Skipping preventable cases raster computation.")
        return out_path
    
    # Read effect size table
    df = pd.read_excel(effect_size_table_path)

    # Get relative odds per unit NDVI.
    odds_ratio = float(df.loc[df['health_indicator'] == 'depression', 'effect_size'].iloc[0])
    #odds_ratio = float(df[df['health_indicator'] == 'depression']['effect_size'].iloc[0])

    # Calculate risk ratio.
    risk_ratio = odds_ratio/(1-p0 + (p0*odds_ratio))

    with rasterio.open(delta_ne_path) as dsrc, rasterio.open(pop_path) as psrc:
        if dsrc.shape != psrc.shape:
            raise ValueError("Delta NE and population rasters must have the same dimensions.")

        pop_nodata = psrc.nodata
        
        profile = dsrc.profile.copy()
        profile.update(dtype='float32', BIGTIFF='YES', nodata=np.nan, compress=compress)

        with rasterio.open(out_path, 'w', **profile) as dst:
            windows = list(dsrc.block_windows(1))
            for _, window in tqdm(windows, desc="Calculating preventable cases"):
                delta_ne = dsrc.read(1, window=window)
                pop = psrc.read(1, window=window)

                if pop_nodata is not None:
                    pop = np.where(pop == pop_nodata, np.nan, pop)  # mask WorldPop's NoData sentinel before it enters arithmetic
                
                # calculate preventable fraction
                pf = np.power(risk_ratio, -10*delta_ne) - 1

                # calculate baseline cases and preventable cases
                bc = prevalence * pop  # baseline cases per pixel
                preventable_cases = bc * pf  # preventable cases per pixel

                # handle NaN values
                preventable_cases = np.where(np.isnan(delta_ne) | np.isnan(pop), np.nan, preventable_cases)

                dst.write(preventable_cases.astype('float32'), 1, window=window)

    #print(f"Preventable cases raster saved to: {out_path}")


def _file_stamp(path):
    """
    Identify a file by absolute path, size and modification time (cheap, no content hashing).
    Used to decide whether a saved zone raster still matches the files it was built from.
    """
    st = os.stat(path)
    return {'path': os.path.abspath(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _zones_vector_path(zone_raster_path):
    """Path of the derived polygons-with-zone_id file that sits beside the zone raster."""
    return os.path.splitext(zone_raster_path)[0] + '_zones.gpkg'


def rasterize_regions_to_reference_grid(
        region_boundary_path,
        reference_path,
        out_zone_raster_path,
        out_lookup_csv_path,
        all_touched=False,
        compress="deflate"
        ):
    """
    Rasterize the urban region polygons onto the reference (population) grid.

    First half of the grid-based zonal statistics. Each pixel receives the integer
    `zone_id` of the region whose polygon contains the pixel centre (0 = outside every region),
    on exactly the reference raster's CRS, extent and resolution. Because the zones now live on
    the same grid as the preventable-cases raster, no raster is ever reprojected. Polygons are
    reprojected only if their CRS differs from the grid's (see below).

    Design points:
      * `zone_id` is a sequential 1..N integer assigned here, not the boundary file's own 'id'
        column (which need not be unique or numeric). The mapping zone_id -> region attributes
        is written to `out_lookup_csv_path`.
      * If the boundary file's CRS differs from the reference raster's, the polygons are
        reprojected to the raster's CRS in memory (geopandas `to_crs`) and the reprojected copy
        is what gets burned, through the derived `_zones.gpkg`. The input boundary file itself is
        never modified. A boundary file with no CRS defined raises an error.
      * A pixel carries one zone_id. If polygons overlap, the later polygon in file order wins
        for the shared pixels, so no pixel is counted in two regions. There is no priority
        rule. (This function does not check whether the boundary file
        contains overlaps.)
      * `all_touched=False` (default): a pixel belongs to a region only if its centre falls
        inside the polygon. (Using ALL_TOUCHED=True gives every polygon at least one pixel, but
	every polygon also gains a ring of edge pixels.) Polygons that end up with no pixel centre
	are handled by the tiny-polygon pass in `aggregate_preventable_cases_by_region`.
      * The raster is written to a temporary name and renamed only when complete, and a
        signature (boundary file, reference raster, all_touched) is saved beside it. An existing
        raster is reused only if that signature still matches, so a changed boundary file or
        population grid triggers a rebuild instead of silently reusing a stale raster.
    """
    
    # define file signature to track source dependencies
    signature_path = out_zone_raster_path + '.signature.json'
    signature = {
        'boundary': _file_stamp(region_boundary_path),
        'reference': _file_stamp(reference_path),
        'all_touched': bool(all_touched),
    }

    # check if zone raster already exists and matches signature
    if os.path.exists(out_zone_raster_path):
        reusable = False
        if os.path.exists(out_lookup_csv_path) and os.path.exists(signature_path):
            with open(signature_path) as fh:
                reusable = json.load(fh) == signature
        if reusable:
            print(f"Zone raster already exists at {out_zone_raster_path} and matches its sources. "
                  "Skipping rasterization.")
            return out_zone_raster_path
        print(f"Zone raster at {out_zone_raster_path} is missing its signature or no longer matches "
              "its sources. Rebuilding.")
    if os.path.exists(signature_path):
        os.remove(signature_path)  # a rebuild in progress mustn't sit beside an old signature

    # load boundaries & check for defined CRS
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
    boundaries = gpd.read_file(region_boundary_path)
    if boundaries.crs is None or ref_crs is None:
        raise ValueError("Both boundary file and reference raster need a defined CRS.")
    
    # reproject boundaries to reference crs in memory
    if not boundaries.crs.equals(pyproj.CRS.from_user_input(ref_crs.to_wkt()), ignore_axis_order=True):
        print(f"Boundary CRS ({boundaries.crs.to_string()}) differs from the reference raster CRS "
              f"({ref_crs.to_string()}). Reprojecting the polygons in memory; the input file is not modified.")
        boundaries = boundaries.to_crs(ref_crs.to_wkt())

    # filter for valid polygon geometries and number them 1 to N
    n_read = len(boundaries)
    boundaries = boundaries[
        boundaries.geometry.notna() &
        boundaries.geometry.geom_type.isin(['Polygon', 'MultiPolygon']) &
        ~boundaries.geometry.is_empty
    ].reset_index(drop=True)
    if len(boundaries) < n_read:
        print(f"Dropped {n_read - len(boundaries):,} null, empty or non-polygon geometries.")
    boundaries['zone_id'] = np.arange(1, len(boundaries) + 1, dtype='int64')  # sequential zone IDs

    # zone_id -> attributes lookup (columns are filled w/ None when boundary file lacks them)
    lookup = pd.DataFrame({'zone_id': boundaries['zone_id'].values})
    lookup['region_id'] = boundaries['id'].values if 'id' in boundaries.columns else lookup['zone_id']
    for out_col, in_col in (('country', 'country'), ('iso3_r250_label', 'iso3'),
                            ('region_name', 'region_name'), ('smod_level', 'smod_level'),
                            ('area_km2', 'area_km2')):
        lookup[out_col] = boundaries[in_col].values if in_col in boundaries.columns else None
    lookup.to_csv(out_lookup_csv_path, index=False)

    # export valid polygons with their zone_ids to vector file for GDAL burn (rasterization)
    zones_vector_path = _zones_vector_path(out_zone_raster_path)
    if os.path.exists(zones_vector_path):
        os.remove(zones_vector_path)
    boundaries[['zone_id', 'geometry']].to_file(zones_vector_path, driver='GPKG')

    # Create zone raster on reference grid and burn zone_id into it.
    partial_path = out_zone_raster_path + '.partial'
    if os.path.exists(partial_path):
        os.remove(partial_path)
    ref_ds = gdal.Open(reference_path)
    out_ds = gdal.GetDriverByName('GTiff').Create(
        partial_path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Int32,
        options=['TILED=YES', 'BIGTIFF=YES', 'SPARSE_OK=TRUE', f'COMPRESS={compress.upper()}']
    )
    if out_ds is None:
        raise RuntimeError(f"Could not create zone raster at {partial_path}")
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    ref_ds = None

    # rasterize polygons onto the grid using zone_id
    vector_ds = ogr.Open(zones_vector_path)
    layer = vector_ds.GetLayer(0)
    rasterize_options = ['ATTRIBUTE=zone_id'] + (['ALL_TOUCHED=TRUE'] if all_touched else [])
    print(f"Rasterizing {len(boundaries):,} regions onto the reference grid...")
    err = gdal.RasterizeLayer(out_ds, [1], layer, options=rasterize_options)
    if err != 0:
        raise RuntimeError(f"gdal.RasterizeLayer failed with error code {err}")
    out_ds.FlushCache()
    out_ds = None
    vector_ds = None

    # finalize raster and write signature file
    os.replace(partial_path, out_zone_raster_path)
    with open(signature_path, 'w') as fh:
        json.dump(signature, fh, indent=2)  # written last; its presence implies everything above finished

    print(f"Zone raster saved to: {out_zone_raster_path}")
    print(f"Zone lookup saved to: {out_lookup_csv_path}")
    return out_zone_raster_path


def _rescue_unmatched_polygons(csrc, zsrc, zones_vector_path, unmatched_ids, sums, counts, window_size):
    """
    Tiny-polygon pass: give regions that own no pixel centre the free pixels they touch.

    A region is "unmatched" when no pixel centre of the population grid falls inside its polygon
    (typically a polygon smaller than one pixel, or a thin sliver). For each such region this
    finds the pixels its outline touches and keeps those that are
      * not owned by any region in the zone raster (zone_id == 0), so no pixel is ever counted
        in two regions;
      * not already given to another rescued region in this pass (first region wins);
      * valid in the preventable-cases raster (not NaN).

    Their cases are added to `sums` and `counts` in place. Only pixels the polygon actually
    touches are used (not its whole bounding box), so a thin sliver does not collect its whole
    bounding box.

    Returns:
        tuple: (number of unmatched regions that received at least one pixel,
                total preventable cases added by this pass).
    """
    
    # skip if grid is not north-up
    transform = csrc.transform
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        print("Tiny-polygon pass skipped: it assumes a north-up grid.")
        return 0, 0.0

    # load polygons for unmatched zone_ids
    n_rows, n_cols = csrc.shape
    zones = gpd.read_file(zones_vector_path)
    zones = zones[zones['zone_id'].isin(unmatched_ids)]

    claimed = set()  # flattened (row * n_cols + col) of pixels already given to a rescued region
    n_rescued, rescued_cases = 0, 0.0
    
    # loop over each unmatched polygon to claim touched pixels
    for zone_id, geom in zip(zones['zone_id'].values, zones.geometry.values):
        # calculate pixel grid indices for polygon bounding box
        xmin, ymin, xmax, ymax = geom.bounds
        col0 = max(int(np.floor((xmin - transform.c) / transform.a)), 0)
        col1 = min(int(np.ceil((xmax - transform.c) / transform.a)), n_cols)
        row0 = max(int(np.floor((ymax - transform.f) / transform.e)), 0)
        row1 = min(int(np.ceil((ymin - transform.f) / transform.e)), n_rows)
        if col1 <= col0 or row1 <= row0:
            continue  # entirely off the grid

        # process touched pixels in chunks
        for row in range(row0, row1, window_size):
            for col in range(col0, col1, window_size):
                window = Window(col, row, min(window_size, col1 - col), min(window_size, row1 - row))
                
                # identify all pixels touched by the polygon
                touched = rasterio.features.rasterize(
                    [(geom, 1)], out_shape=(window.height, window.width),
                    transform=rasterio.windows.transform(window, transform),
                    fill=0, all_touched=True, dtype='uint8'
                ).astype(bool)
                
                # find touched pixels that are currently unassigned
                free = touched & (zsrc.read(1, window=window) == 0)
                if not free.any():
                    continue
                
                # assign valid unassigned pixels to the current zone
                cases = csrc.read(1, window=window)
                rows, cols = np.nonzero(free & np.isfinite(cases))
                for r, c in zip(rows.tolist(), cols.tolist()):
                    key = (row + r) * n_cols + (col + c)
                    if key in claimed:
                        continue
                    claimed.add(key)
                    sums[zone_id] += float(cases[r, c])
                    counts[zone_id] += 1
                    rescued_cases += float(cases[r, c])
        if counts[zone_id] > 0:
            n_rescued += 1

    return n_rescued, rescued_cases


def aggregate_preventable_cases_by_region(
        preventable_cases_raster_path,
        zone_id_raster_path,
        zone_lookup_csv_path,
        out_csv_path,
        window_size=4096,
        rescue_tiny_polygons=True
        ):
    """
    Aggregate preventable cases by urban region blockwise on the population grid.

    This is the second half of grid-based zonal stats. The preventable-cases raster and
    the zone raster from `rasterize_regions_to_reference_grid` share one grid, so both are read
    in matching windows and the cases are summed per zone_id with np.bincount. No raster is
    reprojected and there is no per-polygon masking.

    Regions that own no pixel centre get none. We give a tiny-polygon pass (`rescue_tiny_polygons=True`
    argument) in `_rescue_unmatched_polygons` using the pixels the polygon touches and only
    pixels no other region owns. Set it to False to leave such regions out. The pass reads the
    derived `_zones.gpkg` beside the zone raster.

    Regions with no valid pixels are omitted.
    """
    
    # initialize arrays for zone aggregation
    lookup = pd.read_csv(zone_lookup_csv_path)
    n_zones = int(lookup['zone_id'].max())
    sums = np.zeros(n_zones + 1, dtype=np.float64)
    counts = np.zeros(n_zones + 1, dtype=np.int64)
    owned = np.zeros(n_zones + 1, dtype=bool)  # zone has >=1 pixel centre on the grid (valid or nodata)
    n_rescued, rescued_cases = 0, 0.0

    # verify rasters share the same shape, transform, and crs
    with rasterio.open(preventable_cases_raster_path) as csrc, rasterio.open(zone_id_raster_path) as zsrc:
        if csrc.shape != zsrc.shape:
            raise ValueError(
                f"Preventable-cases raster {csrc.shape} and zone raster {zsrc.shape} differ in shape; "
                "rebuild the zone raster on the population grid."
            )
        if not np.allclose(tuple(csrc.transform)[:6], tuple(zsrc.transform)[:6], rtol=0, atol=1e-9):
            raise ValueError("Preventable-cases raster and zone raster are not on the same grid (geotransforms differ).")
        if csrc.crs != zsrc.crs:
            raise ValueError("Preventable-cases raster and zone raster have different CRS.")

        # define processing windows
        n_rows, n_cols = csrc.shape
        windows = [
            Window(col, row, min(window_size, n_cols - col), min(window_size, n_rows - row))
            for row in range(0, n_rows, window_size)
            for col in range(0, n_cols, window_size)
        ]

        # aggregate preventable cases blockwise
        for window in tqdm(windows, desc="Summing preventable cases by region"):
            zones = zsrc.read(1, window=window)
            in_zone = zones > 0
            if not in_zone.any():
                continue

            # extract zone ids and cases for valid pixels
            ids = zones[in_zone]
            values = csrc.read(1, window=window)[in_zone]
            finite = np.isfinite(values)  # NaN = nodata (no population or no NDVI)

            # sum cases and count valid pixels per zone using np.bincount
            uniq, inverse = np.unique(ids, return_inverse=True)
            inverse = inverse.ravel()
            if uniq[-1] > n_zones:
                raise ValueError(
                    f"Zone raster holds zone_id {int(uniq[-1])} but the lookup stops at {n_zones}; "
                    "the zone raster and lookup are from different runs."
                )
            owned[uniq] = True
            if finite.any():
                sums[uniq] += np.bincount(inverse[finite], weights=values[finite].astype(np.float64),
                                          minlength=uniq.size)
                counts[uniq] += np.bincount(inverse[finite], minlength=uniq.size)

        # identify and rescue regions with no pixel centers
        unmatched = np.flatnonzero(~owned[1:]) + 1
        print(f"{unmatched.size:,} regions contain no pixel centre of the population grid "
              "(for example, polygons smaller than a pixel).")
        if rescue_tiny_polygons and unmatched.size:
            zones_vector_path = _zones_vector_path(zone_id_raster_path)
            if os.path.exists(zones_vector_path):
                n_rescued, rescued_cases = _rescue_unmatched_polygons(
                    csrc, zsrc, zones_vector_path, unmatched, sums, counts, window_size)
                print(f"Tiny-polygon pass: {n_rescued:,} of them received at least one free pixel they "
                      f"touch (adding {rescued_cases:,.0f} preventable cases); "
                      f"{unmatched.size - n_rescued:,} still have none (every pixel they touch is owned "
                      "by another region, is nodata, or is off the grid).")
            else:
                print(f"Tiny-polygon pass skipped: {zones_vector_path} not found.")

    # merge aggregated totals with region lookup attributes
    present = np.flatnonzero(counts)
    df = pd.DataFrame({
        'zone_id': present,
        'n_valid_pixels': counts[present],
        'total_preventable_cases': sums[present],
    })
    df = df.merge(lookup, on='zone_id', how='left', validate='one_to_one')
    df = df[['region_id', 'country', 'iso3_r250_label', 'region_name', 'smod_level', 'area_km2',
             'n_valid_pixels', 'total_preventable_cases']]

    print(f"{len(df):,} of {len(lookup):,} regions received at least one valid pixel; "
          f"{len(lookup) - len(df):,} had none and are omitted.")
    print(f"Regions with negative total preventable cases: {int((df['total_preventable_cases'] < 0).sum()):,}; "
          f"total across regions: {df['total_preventable_cases'].sum():,.0f}")

    # round totals and save to csv
    df['total_preventable_cases'] = df['total_preventable_cases'].round(0).astype(int)
    df.to_csv(out_csv_path, index=False)
    print(f"Regional preventable cases summary written to: {out_csv_path}")

    return out_csv_path


def apply_country_costs(
        regional_cases_csv_path,
        health_cost_rate_path,
        out_country_csv_path,
        iso3_col='iso3_r250_label',
        cost_col='price_usd_2019'
        ):
    """
    Apply country-specific treatment costs (USD 2019 per case) to aggregated preventable cases.

    Regional cases are summed to the country level and joined to the cost table on ISO3.
    I use left join from the cases side, so every country that has preventable cases
    stays in the output. Countries with no price get NaN (empty) cost_savings rather than
    being dropped.
    """

    # read input data
    regional_df = pd.read_csv(regional_cases_csv_path)
    cost_df = pd.read_csv(health_cost_rate_path)

    # check expected columns exist
    for name, df, cols in (
            ('regional cases', regional_df, [iso3_col, 'country', 'total_preventable_cases']),
            ('treatment cost', cost_df, [iso3_col, cost_col])
            ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} table is missing column(s) {missing}. Found: {list(df.columns)}")

    # Normalise ISO3 codes so that stray whitespace or case cannot break the join.
    for df in (regional_df, cost_df):
        df[iso3_col] = df[iso3_col].astype('string').str.strip().str.upper()

    if regional_df[iso3_col].isna().all():
        raise ValueError(
            f"No ISO3 codes found in '{iso3_col}' of {regional_cases_csv_path}. "
            "Check that the urban boundaries file has an 'iso3' column."
        )

    # cost table must have exactly one price per ISO3, else the merge duplicates cases
    cost_df = cost_df.dropna(subset=[iso3_col])
    duplicated = cost_df.loc[cost_df[iso3_col].duplicated(keep=False), iso3_col].unique().tolist()
    if duplicated:
        raise ValueError(f"Treatment cost table has multiple rows for ISO3 code(s): {duplicated}")

    # sum preventable cases for each country (keyed on ISO3)
    has_iso3 = regional_df[iso3_col].notna()
    country_sums = regional_df[has_iso3].groupby(iso3_col, as_index=False).agg(
        country=('country', 'first'),
        total_preventable_cases=('total_preventable_cases', 'sum')
    )

    # Regions with no ISO3 code are kept (grouped by country name) instead of silently dropped.
    if (~has_iso3).any():
        print(f"WARNING: {int((~has_iso3).sum())} region(s) have no ISO3 code; keeping them with no cost.")
        no_iso3_sums = regional_df[~has_iso3].groupby('country', dropna=False, as_index=False).agg(
            total_preventable_cases=('total_preventable_cases', 'sum')
        )
        no_iso3_sums[iso3_col] = pd.NA
        country_sums = pd.concat([country_sums, no_iso3_sums], ignore_index=True)

    # left-join the per-case cost on ISO3 so that no country is dropped
    merged = country_sums.merge(
        cost_df[[iso3_col, cost_col]].rename(columns={cost_col: 'cost_per_case'}),
        on=iso3_col,
        how='left',
        validate='many_to_one'
    )

    # report countries that have preventable cases but no price
    no_price = merged.loc[merged['cost_per_case'].isna(), iso3_col].fillna('<missing ISO3>').tolist()
    print(f"Cost matched for {len(merged) - len(no_price)} of {len(merged)} countries; "
          f"{len(no_price)} without a price (cost_savings left empty).")

    merged['cost_savings'] = merged['total_preventable_cases'] * merged['cost_per_case']  # cost savings (NaN where no price is available)

    merged[['country', iso3_col, 'total_preventable_cases', 'cost_savings']].to_csv(
        out_country_csv_path, index=False, float_format='%.2f'
    )  # output CSV with desired columns
    print(f"Country-level cost savings saved to: {out_country_csv_path}")

    return out_country_csv_path
