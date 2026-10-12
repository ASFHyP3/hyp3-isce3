"""Stage the NISAR DEM, kept apart from process.py so the focus workflow can use it without the InSAR stack."""

import earthaccess
from nisar.workflows import stage_dem
from osgeo import gdal


def get_dem(reference_path: str, subset: list[float] | None = None) -> str:
    """Download the NISAR DEM tiles covering the scene and mosaic them into one GeoTIFF.

    Args:
        reference_path: Path of the reference scene.
        subset: Optional AOI [lon_min, lat_min, lon_max, lat_max]; when set, the DEM
            is fetched over the AOI instead of the whole frame (the 1-degree buffer
            below covers the crop margin).

    Returns:
        dem_path: Path of the mosaicked DEM file.
    """
    short_name = 'NISAR_DEM'
    if subset:
        bbox = tuple(subset)
    else:
        poly, _ = stage_dem.determine_polygon(reference_path, bbox=None, bbox_epsg='4326')
        bbox = poly.bounds
    buf = 1.0  # degrees padded around the AOI or frame so the mosaic covers the crop margin
    bbox = (bbox[0] - buf, bbox[1] - buf, bbox[2] + buf, bbox[3] + buf)
    results = earthaccess.search_data(short_name=short_name, bounding_box=bbox)
    files = sorted(earthaccess.download(results))
    folder = files[0].parent
    files = [f.resolve() for f in files if '.tif' in f.name]
    # Near the poles the search also returns polar-stereographic (EPSG:3413/3031) copies of the same
    # DEM. The EPSG:4326 tiles cover the globe on their own, and BuildVRT can't mix projections.
    files = [f for f in files if gdal.Open(str(f)).GetSpatialRef().GetAuthorityCode(None) == '4326']
    if not files:
        raise RuntimeError(f'No EPSG:4326 NISAR DEM tiles found for bbox {bbox}')

    output_raster = folder / 'dem.tif'
    vrt_dataset = gdal.BuildVRT(str(folder / 'mosaic.vrt'), files)
    vrt_dataset = None
    vrt_dataset = folder / 'mosaic.vrt'
    # gdal.Warp(str(output_raster), vrt_dataset, dstSRS="EPSG:4326")
    gdal.Warp(str(output_raster), vrt_dataset)

    if output_raster.exists():
        return str(output_raster)
    else:
        raise RuntimeError('DEM could not be downloaded')
