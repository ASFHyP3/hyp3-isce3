"""Mosaic NISAR DEM and water mask tiles, kept apart from process.py so the focus workflow can use it without the InSAR stack."""

import earthaccess
from nisar.workflows import stage_dem
from osgeo import gdal


def mosaic_tiles(short_name: str, reference_path: str, subset: list[float] | None, name: str) -> str:
    """Download the tiles of a NISAR raster collection covering the scene and mosaic them into one GeoTIFF.

    Args:
        short_name: CMR collection short name (NISAR_DEM or NISAR_WATERMASK).
        reference_path: Path of the reference scene.
        subset: Optional AOI [lon_min, lat_min, lon_max, lat_max]; when set, the tiles are
            fetched over the AOI instead of the whole frame (the 1-degree buffer below covers
            the crop margin).
        name: File name of the mosaic.

    Returns:
        Path of the mosaicked file.
    """
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
    # tiles. The EPSG:4326 tiles cover the globe on their own, and BuildVRT can't mix projections.
    files = [f for f in files if gdal.Open(str(f)).GetSpatialRef().GetAuthorityCode(None) == '4326']
    if not files:
        raise RuntimeError(f'No EPSG:4326 {short_name} tiles found for bbox {bbox}')

    output_raster = folder / name
    vrt_path = folder / 'mosaic.vrt'
    gdal.BuildVRT(str(vrt_path), files).FlushCache()
    gdal.Warp(str(output_raster), str(vrt_path))

    if output_raster.exists():
        return str(output_raster)
    else:
        raise RuntimeError(f'{name} could not be downloaded')
