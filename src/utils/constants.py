"""Shared constants for TOFU probing and directional intervention."""

FORGET_SPLIT = "forget10"

ABLATION_METHOD_HOOKS = "hooks"
ABLATION_METHOD_ORTHOGONALISATION = "orthogonalisation"
ABLATION_METHOD_STEER = "steer"

DIRECTION_SOURCE_REFUSAL = "refusal"
DIRECTION_SOURCE_CONFABULATION = "confabulation"
DIRECTION_SOURCE_CONFIG_KEYS = {
    DIRECTION_SOURCE_REFUSAL: "refusal_direction",
    DIRECTION_SOURCE_CONFABULATION: "confabulation_direction",
}
