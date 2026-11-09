from __future__ import annotations

from rivwq.gee import collections, masks, stats


def stats_reducer(config: dict):
    import ee

    reducer = None
    for name in config["reducers"]:
        part = getattr(ee.Reducer, name)()
        reducer = part if reducer is None else reducer.combine(part, "", True)
    return reducer


def scene_feature_factory(config: dict, point, geometries: dict):
    import ee

    static_water = masks.connected_water(config, point)
    tags = stats.cloud_tags(config)
    bands = stats.band_list(config)
    scale = config["scale_m"]
    reducer = stats_reducer(config)
    circles = {name: ee.Geometry(info, None, False) for name, info in geometries.items()}
    properties = list(config["scene_properties"])

    def compute(image):
        image = ee.Image(image)
        projection = image.select(config["bands"]["footprint"]).projection()
        footprint = image.select(config["bands"]["footprint"]).mask()
        valid = masks.cloud_valid_images(image, config, tags)
        data = image.select(bands)
        feature = ee.Feature(None, image.toDictionary(properties + ["has_cloud_score"]))
        feature = feature.set("system_index", image.get("system:index"))
        for erosion in config["masks"]["erosion_pixels"]:
            water = masks.eroded(
                static_water, projection, erosion, config["masks"]["erosion_kernel"]
            ).And(footprint)
            layers = []
            counted = []
            for tag in tags:
                selected = water.And(valid[tag])
                layers.append(
                    data.updateMask(selected).rename([f"{tag}__{band}" for band in bands])
                )
                counted.append(ee.Image.constant(1).updateMask(selected).rename(f"{tag}__valid"))
            counted.append(ee.Image.constant(1).updateMask(water).rename("total"))
            stack = ee.Image.cat(layers)
            counts = ee.Image.cat(counted)
            for roi, circle in circles.items():
                values = stack.reduceRegion(
                    reducer=reducer,
                    geometry=circle,
                    crs=projection,
                    scale=scale,
                    maxPixels=config["max_pixels"],
                    tileScale=config["tile_scale"],
                )
                tallies = counts.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=circle,
                    crs=projection,
                    scale=scale,
                    maxPixels=config["max_pixels"],
                    tileScale=config["tile_scale"],
                )
                old_counts = [f"{tag}__valid" for tag in tags] + ["total"]
                new_counts = [stats.count_name(roi, erosion, tag) for tag in tags] + [
                    stats.total_name(roi, erosion)
                ]
                old_stats = [
                    f"{tag}__{band}_{reducer_name}"
                    for tag in tags
                    for band in bands
                    for reducer_name in stats.REDUCER_SUFFIX
                ]
                new_stats = [
                    stats.column_name(roi, erosion, tag, band, suffix)
                    for tag in tags
                    for band in bands
                    for suffix in stats.REDUCER_SUFFIX.values()
                ]
                feature = feature.set(tallies.rename(old_counts, new_counts))
                feature = feature.set(values.rename(old_stats, new_stats))
        return feature

    return compute


def scene_ids(config: dict, point, start: str, end: str) -> list[str]:
    radius = max(config["roi"]["variants"].values())
    scenes = collections.joined_scenes(config, point.buffer(radius), start, end)
    return sorted(scenes.aggregate_array("system:index").getInfo())


def scene_batch_collection(
    config: dict, point, start: str, end: str, ids: list[str], geometries: dict
):
    import ee

    radius = max(config["roi"]["variants"].values())
    scenes = collections.joined_scenes(config, point.buffer(radius), start, end)
    chosen = scenes.filter(ee.Filter.inList("system:index", ids))
    compute = scene_feature_factory(config, point, geometries)
    return ee.FeatureCollection(chosen.map(compute))


def roi_area_table(config: dict, point, epsg: str) -> dict:
    import ee

    water = masks.connected_water(config, point)
    projection = ee.Projection(epsg).atScale(config["scale_m"])
    area = ee.Image.pixelArea()
    result = {}
    for erosion in config["masks"]["erosion_pixels"]:
        mask = masks.eroded(water, projection, erosion, config["masks"]["erosion_kernel"])
        for roi, radius in config["roi"]["variants"].items():
            total = (
                area.updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=point.buffer(radius),
                    crs=projection,
                    scale=config["scale_m"],
                    maxPixels=config["max_pixels"],
                    tileScale=config["tile_scale"],
                )
                .get("area")
            )
            result[f"{roi}__{stats.erosion_tag(erosion)}"] = total
    return ee.Dictionary(result)


def water_polygons(config: dict, point) -> dict:

    water = masks.connected_water(config, point)
    radius = max(config["roi"]["variants"].values())
    vectors = (
        water.toInt()
        .selfMask()
        .reduceToVectors(
            geometry=point.buffer(radius),
            scale=config["masks"]["gsw_scale_m"],
            crs=water.projection(),
            geometryType="polygon",
            eightConnected=False,
            maxPixels=config["max_pixels"],
            tileScale=config["tile_scale"],
        )
    )
    polygon = vectors.geometry(maxError=1)
    return {
        name: polygon.intersection(point.buffer(radius_m), 1)
        for name, radius_m in config["roi"]["variants"].items()
    }
