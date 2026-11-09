from __future__ import annotations


def connected_water(config: dict, point):
    import ee

    settings = config["masks"]
    occurrence = (
        ee.Image(config["collections"]["surface_water"])
        .select(config["bands"]["occurrence"])
        .unmask(0)
    )
    water = occurrence.gte(settings["water_occurrence_min"]).selfMask()
    radius = max(config["roi"]["variants"].values())
    circle = point.buffer(radius)
    cost = water.clip(circle)
    near = ee.Image(0).paint(
        ee.FeatureCollection([ee.Feature(point.buffer(settings["gauge_seed_distance_m"]))]), 1
    )
    seed = water.updateMask(near)
    reached = cost.cumulativeCost(
        source=seed.unmask(0), maxDistance=settings["connectivity_max_path_m"]
    )
    return reached.mask().clip(circle).reproject(occurrence.projection()).rename("water")


def eroded(mask, projection, pixels: int, kernel_type: str):
    import ee

    resampled = mask.reproject(projection)
    if pixels == 0:
        return resampled
    kernel = ee.Kernel.plus(radius=pixels, units="pixels")
    if kernel_type != "plus":
        raise ValueError(f"unsupported erosion kernel: {kernel_type}")
    return resampled.focalMin(kernel=kernel)


def cloud_valid_images(image, config: dict, tags: list[str]) -> dict:

    settings = config["masks"]
    score = image.select(config["bands"]["cloud_score"])
    classification = image.select(config["bands"]["scene_classification"])
    classes = settings["scl_valid_classes"]
    valid = {}
    for threshold, tag in zip(settings["cloud_score_thresholds"], tags[:-1], strict=True):
        valid[tag] = score.gte(threshold)
    valid[tags[-1]] = classification.remap(classes, [1] * len(classes), 0).eq(1)
    return valid
