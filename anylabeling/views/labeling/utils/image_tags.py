from anylabeling.views.labeling.logger import logger


def normalize_image_tag(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or "\r" in value or "\n" in value:
        return None
    return value


def normalize_image_tags(value, source=None):
    if not isinstance(value, list):
        if source:
            logger.warning(f"Invalid image tags in {source}: expected a list")
        return []

    tags = []
    seen = set()
    invalid_types = 0
    invalid_text = 0
    duplicates = 0
    for item in value:
        tag = normalize_image_tag(item)
        if tag is None:
            if isinstance(item, str):
                invalid_text += 1
            else:
                invalid_types += 1
            continue
        if tag in seen:
            duplicates += 1
            continue
        tags.append(tag)
        seen.add(tag)

    issues = []
    if invalid_types:
        issues.append(f"{invalid_types} non-string item(s)")
    if invalid_text:
        issues.append(f"{invalid_text} empty or multiline item(s)")
    if duplicates:
        issues.append(f"{duplicates} duplicate item(s)")
    if source and issues:
        logger.warning(f"Invalid image tags in {source}: {', '.join(issues)}")
    return tags
