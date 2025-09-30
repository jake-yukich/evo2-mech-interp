import random

def overlaps(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    """Helper, check if the interval (start, end) overlaps with any in intervals."""
    for used_start, used_end in intervals:
        if not (end <= used_start or start >= used_end):
            return True
    return False

# TODO: shrink this function 
def sample_genome(
    text: str,
    sample_region_length: int,
    coverage_fraction: float | None = None,
    num_samples: int | None = None,
    max_attempts: int = 10000,
) -> list[str]:
    """
    Randomly sample non-overlapping regions from the input text until at least
    `coverage_fraction` (default 5%) of the text is covered OR `num_samples` (default 10) samples are obtained
    .
    Returns a list of non-overlapping sampled regions.
    """
    if (coverage_fraction is not None) and (num_samples is not None):
        raise ValueError("Specify either coverage_fraction or num_samples, not both.")
    if (coverage_fraction is None) and (num_samples is None):
        raise ValueError("Specify either coverage_fraction or num_samples.")
    text_length = len(text)

    if text_length < sample_region_length:
        print(
            f"Warning: Text is shorter than sample_region_length ({text_length} < {sample_region_length})"
        )
        return []

    target_coverage = (
        int(text_length * coverage_fraction)
        if coverage_fraction is not None
        else num_samples * sample_region_length
    )
    covered_bp = 0
    used_intervals = []
    regions = []
    attempts = 0

    if coverage_fraction is not None:
        print(f"Sampling to cover {target_coverage} bp ({coverage_fraction * 100:.1f}% of text)")
    else:
        print(f"Sampling {num_samples} regions of {sample_region_length} bp each")

    while covered_bp < target_coverage and attempts < max_attempts:
        start_pos = random.randint(0, text_length - sample_region_length, )
        end_pos = start_pos + sample_region_length

        if not overlaps(start_pos, end_pos, used_intervals):
            region = text[start_pos:end_pos]
            regions.append(region)
            used_intervals.append((start_pos, end_pos))
            covered_bp += sample_region_length
        attempts += 1

        # If we've exhausted all possible non-overlapping regions, break
        if len(used_intervals) >= (text_length // sample_region_length):
            break

    if coverage_fraction is not None and covered_bp < target_coverage:
        print(
            f"Warning: Only able to cover {covered_bp} bp out of requested {target_coverage} bp ({coverage_fraction * 100:.1f}% of text)."
        )

    return regions