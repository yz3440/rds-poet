"""Keep the two audio observations and any lead-station words distinguishable."""

NEARBY = "\nNearby channel: "
TRANSCRIPT = "\nWhisper transcript: "


def pair_observations(lead, neighbor):
    acoustic, marker, speech = lead.partition(TRANSCRIPT)
    return acoustic + NEARBY + neighbor.splitlines()[0] + marker + speech


def split_cue(cue):
    acoustic, _, speech = cue.partition(TRANSCRIPT)
    lead, _, neighbor = acoustic.partition(NEARBY)
    return lead, neighbor, speech
