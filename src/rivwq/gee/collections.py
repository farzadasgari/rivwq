from __future__ import annotations


def joined_scenes(config: dict, geometry, start: str, end: str):
    import ee

    cloud_band = config["bands"]["cloud_score"]
    reflectance = (
        ee.ImageCollection(config["collections"]["surface_reflectance"])
        .filterBounds(geometry)
        .filterDate(start, end)
    )
    scores = (
        ee.ImageCollection(config["collections"]["cloud_score"])
        .filterBounds(geometry)
        .filterDate(start, end)
    )
    same_index = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.Join.saveFirst("cloud_match").apply(reflectance, scores, same_index)
    absent = ee.Image.constant(0).rename(cloud_band).updateMask(ee.Image.constant(0))

    def attach(image):
        image = ee.Image(image)
        match = image.get("cloud_match")
        score = ee.Image(
            ee.Algorithms.If(match, ee.Image(match).select([cloud_band]), absent)
        ).rename(cloud_band)
        return image.addBands(score).set("has_cloud_score", ee.Algorithms.If(match, 1, 0))

    return ee.ImageCollection(joined).map(attach)
