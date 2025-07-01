def apply_option_mapping(value, mapping):
    """Map options to their corresponding values."""
    if isinstance(value, str):
        return mapping.get(value, value)
    elif isinstance(value, list):
        return [mapping.get(v, v) for v in value]
    return value


def value_contains_deleted_options(value, deleted_options):
    """Check if the value includes any deleted options."""
    if isinstance(value, str):
        return value in deleted_options
    elif isinstance(value, list):
        return any(v in deleted_options for v in value)
    return False


def get_default_value(comp_type, options):
    """Return the default value based on component type."""
    if comp_type == "QRadioButton" and options:
        return options[0]
    elif comp_type == "QCheckBox":
        return []
    elif comp_type == "QComboBox":
        return None
    return ""


def get_real_modified_options(old_options, new_options, common_options):
    """Identify truly modified options excluding common ones."""
    modified = {}

    if len(old_options) == len(new_options):
        for i in range(len(old_options)):
            old_opt = old_options[i]
            new_opt = new_options[i]
            if old_opt != new_opt and old_opt not in common_options:
                modified[old_opt] = new_opt

    return modified
