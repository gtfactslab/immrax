import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.comparison import IntervalRelation, interval_compare

# =============================================================================
# Interval Definitions
# =============================================================================

# Shape (1,) intervals
INTERVALS_1D = {
    "1_2": irx.interval(jnp.array([1.0]), jnp.array([2.0])),
    "1_3": irx.interval(jnp.array([1.0]), jnp.array([3.0])),
    "1_4": irx.interval(jnp.array([1.0]), jnp.array([4.0])),
    "2_3": irx.interval(jnp.array([2.0]), jnp.array([3.0])),
    "2_4": irx.interval(jnp.array([2.0]), jnp.array([4.0])),
    "3_4": irx.interval(jnp.array([3.0]), jnp.array([4.0])),
    "3_5": irx.interval(jnp.array([3.0]), jnp.array([5.0])),
    # Singletons
    "2_2": irx.interval(jnp.array([2.0]), jnp.array([2.0])),
    "2.5_2.5": irx.interval(jnp.array([2.5]), jnp.array([2.5])),
    "3_3": irx.interval(jnp.array([3.0]), jnp.array([3.0])),
    # Empty (lower > upper)
    "empty_3_1": irx.interval(jnp.array([3.0]), jnp.array([1.0])),
    "empty_5_2": irx.interval(jnp.array([5.0]), jnp.array([2.0])),
}

# Scalar intervals (shape ())
INTERVALS_SCALAR = {
    "1_2": irx.interval(jnp.array(1.0), jnp.array(2.0)),
    "5_6": irx.interval(jnp.array(5.0), jnp.array(6.0)),
}

# Shape (3,) intervals
INTERVALS_3D = {
    "123_234": irx.interval(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 3.0, 4.0])),
    "567_678": irx.interval(jnp.array([5.0, 6.0, 7.0]), jnp.array([6.0, 7.0, 8.0])),
}

# Shape (4,) intervals
INTERVALS_4D = {
    "1234_2345": irx.interval(
        jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([2.0, 3.0, 4.0, 5.0])
    ),
}

# =============================================================================
# Parametrized Fixtures
# =============================================================================


@pytest.fixture(
    params=[
        (INTERVALS_1D["1_2"], INTERVALS_1D["3_4"], IntervalRelation.PRECEDES),
        (INTERVALS_1D["1_2"], INTERVALS_1D["2_3"], IntervalRelation.MEETS),
        (INTERVALS_1D["1_3"], INTERVALS_1D["2_4"], IntervalRelation.OVERLAPS),
        (INTERVALS_1D["1_2"], INTERVALS_1D["1_3"], IntervalRelation.STARTS),
        (INTERVALS_1D["2_3"], INTERVALS_1D["1_4"], IntervalRelation.DURING),
        (INTERVALS_1D["2_3"], INTERVALS_1D["1_3"], IntervalRelation.FINISHES),
        (INTERVALS_1D["1_2"], INTERVALS_1D["1_2"], IntervalRelation.EQUAL),
        (INTERVALS_1D["2_2"], INTERVALS_1D["3_3"], IntervalRelation.PRECEDES),
        (INTERVALS_1D["2_2"], INTERVALS_1D["2_2"], IntervalRelation.EQUAL),
        (INTERVALS_1D["2_2"], INTERVALS_1D["2_3"], IntervalRelation.STARTS),
        (INTERVALS_1D["3_3"], INTERVALS_1D["2_3"], IntervalRelation.FINISHES),
        (INTERVALS_1D["2.5_2.5"], INTERVALS_1D["2_3"], IntervalRelation.DURING),
        (INTERVALS_1D["3_5"], INTERVALS_1D["1_3"], IntervalRelation.MET_BY),
    ]
)
def allen_relation_pair(request):
    """Parametrized fixture for the 7 unique Allen relations."""
    return request.param


@pytest.fixture(
    params=[
        (
            INTERVALS_SCALAR["1_2"],
            INTERVALS_SCALAR["5_6"],
            IntervalRelation.PRECEDES,
            (),
        ),
        (
            INTERVALS_SCALAR["1_2"],
            INTERVALS_3D["567_678"],
            IntervalRelation.PRECEDES,
            (3,),
        ),
        (
            INTERVALS_3D["123_234"],
            INTERVALS_3D["567_678"],
            IntervalRelation.PRECEDES,
            (3,),
        ),
    ]
)
def broadcasting_pair(request):
    """Parametrized fixture for broadcasting test cases."""
    return request.param


# =============================================================================
# Test Functions: Standard Allen Relations
# =============================================================================


def test_interval_compare_allen_relations(allen_relation_pair):
    """Tests interval_compare for all 7 unique Allen relations."""
    a, b, expected = allen_relation_pair
    result = interval_compare(a, b)

    assert isinstance(result, IntervalRelation), (
        f"Expected IntervalRelation, got {type(result).__name__}"
    )
    assert result.shape == a.shape, (
        f"Result shape {result.shape} does not match input shape {a.shape}"
    )
    assert result.shape == b.shape, (
        f"Result shape {result.shape} does not match input shape {b.shape}"
    )
    assert jnp.all(result == expected), f"Expected {expected}, got {result}"


# =============================================================================
# Test Functions: Broadcasting
# =============================================================================


def test_interval_compare_broadcasting(broadcasting_pair):
    """Tests interval_compare broadcasting behavior."""
    a, b, expected, expected_shape = broadcasting_pair
    result = interval_compare(a, b)

    assert isinstance(result, IntervalRelation), (
        f"Expected IntervalRelation, got {type(result).__name__}"
    )
    assert result.shape == expected_shape, (
        f"Result shape {result.shape} does not match expected shape {expected_shape}"
    )
    assert jnp.all(result == expected), f"Expected {expected}, got {result}"


def test_interval_compare_incompatible_shapes_raises():
    """Tests that comparing intervals with incompatible shapes raises an error."""
    with pytest.raises(Exception):
        interval_compare(INTERVALS_3D["123_234"], INTERVALS_4D["1234_2345"])


def test_interval_compare_broadcasting_matches_elementwise():
    """Tests that JAX broadcasting matches manual elementwise comparison."""
    a, b = INTERVALS_3D["123_234"], INTERVALS_3D["567_678"]
    broadcasted_result = interval_compare(a, b)

    elementwise_results = []
    for i in range(3):
        a_i = irx.interval(jnp.array([a.lower[i]]), jnp.array([a.upper[i]]))
        b_i = irx.interval(jnp.array([b.lower[i]]), jnp.array([b.upper[i]]))
        elementwise_results.append(interval_compare(a_i, b_i).value[0])

    elementwise_result = IntervalRelation(jnp.array(elementwise_results))

    assert jnp.all(broadcasted_result.value == elementwise_result.value), (
        f"Broadcasted {broadcasted_result.value} != elementwise {elementwise_result.value}"
    )


def test_interval_compare_broadcasting_mixed_relations():
    """Tests broadcasting with different relations per element."""
    a = irx.interval(jnp.array([1.0, 2.0, 1.0]), jnp.array([2.0, 4.0, 3.0]))
    b = irx.interval(jnp.array([5.0, 3.0, 1.0]), jnp.array([6.0, 5.0, 3.0]))
    result = interval_compare(a, b)

    expected_values = jnp.array(
        [
            int(IntervalRelation.PRECEDES),
            int(IntervalRelation.OVERLAPS),
            int(IntervalRelation.EQUAL),
        ]
    )

    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert jnp.all(result.value == expected_values), (
        f"Expected {expected_values}, got {result.value}"
    )


# =============================================================================
# Tests for natif with lt
# =============================================================================


def test_natif_lt_with_precedes_mask():
    """Tests that natif-transformed lt function works with Intervals.

    Defines a comparison function using irx.lt with PRECEDES mask,
    transforms it with natif, and verifies correct boolean results
    for interval pairs that do and do not satisfy the PRECEDES relation.
    """

    def compare_lt(a, b):
        return irx.lt(a, b, relation_mask=IntervalRelation.PRECEDES)

    compare_lt_intervals = irx.natif(compare_lt)

    # Test case 1: [1,2] PRECEDES [3,4] - should return True
    a_precedes = INTERVALS_1D["1_2"]
    b_precedes = INTERVALS_1D["3_4"]
    result_precedes = compare_lt_intervals(a_precedes, b_precedes)
    assert jnp.all(result_precedes), (
        f"Expected True for [1,2] PRECEDES [3,4], got {result_precedes}"
    )

    # Test case 2: [1,2] MEETS [2,3] - should return False (not PRECEDES)
    a_meets = INTERVALS_1D["1_2"]
    b_meets = INTERVALS_1D["2_3"]
    result_meets = compare_lt_intervals(a_meets, b_meets)
    assert jnp.all(~result_meets), (
        f"Expected False for [1,2] PRECEDES [2,3], got {result_meets}"
    )

    # Test case 3: [2,3] DURING [1,4] - should return False (not PRECEDES)
    a_during = INTERVALS_1D["2_3"]
    b_during = INTERVALS_1D["1_4"]
    result_during = compare_lt_intervals(a_during, b_during)
    assert jnp.all(~result_during), (
        f"Expected False for [2,3] PRECEDES [1,4], got {result_during}"
    )

    # Test case 4: [1,2] EQUAL [1,2] - should return False (not PRECEDES)
    a_equal = INTERVALS_1D["1_2"]
    b_equal = INTERVALS_1D["1_2"]
    result_equal = compare_lt_intervals(a_equal, b_equal)
    assert jnp.all(~result_equal), (
        f"Expected False for [1,2] PRECEDES [1,2], got {result_equal}"
    )

    # Test case 5: [3,4] PRECEDED_BY [1,2] - should return False (converse)
    a_preceded_by = INTERVALS_1D["3_4"]
    b_preceded_by = INTERVALS_1D["1_2"]
    result_preceded_by = compare_lt_intervals(a_preceded_by, b_preceded_by)
    assert jnp.all(~result_preceded_by), (
        f"Expected False for [3,4] PRECEDES [1,2], got {result_preceded_by}"
    )


# =============================================================================
# Tests for IntervalRelation Type
# =============================================================================


class TestIntervalRelationType:
    """Tests for the IntervalRelation class itself."""

    def test_relation_constants_exist(self):
        """Tests that all 13 relation constants are defined."""
        relations = [
            IntervalRelation.PRECEDES,
            IntervalRelation.MEETS,
            IntervalRelation.OVERLAPS,
            IntervalRelation.STARTS,
            IntervalRelation.DURING,
            IntervalRelation.FINISHES,
            IntervalRelation.EQUAL,
            IntervalRelation.PRECEDED_BY,
            IntervalRelation.MET_BY,
            IntervalRelation.OVERLAPPED_BY,
            IntervalRelation.STARTED_BY,
            IntervalRelation.CONTAINS,
            IntervalRelation.FINISHED_BY,
        ]
        for rel in relations:
            assert isinstance(rel, IntervalRelation)

    def test_relation_constants_unique(self):
        """Tests that all 13 relation constants have unique values."""
        relations = [
            IntervalRelation.PRECEDES,
            IntervalRelation.MEETS,
            IntervalRelation.OVERLAPS,
            IntervalRelation.STARTS,
            IntervalRelation.DURING,
            IntervalRelation.FINISHES,
            IntervalRelation.EQUAL,
            IntervalRelation.PRECEDED_BY,
            IntervalRelation.MET_BY,
            IntervalRelation.OVERLAPPED_BY,
            IntervalRelation.STARTED_BY,
            IntervalRelation.CONTAINS,
            IntervalRelation.FINISHED_BY,
        ]
        values = [int(r) for r in relations]
        assert len(values) == len(set(values))

    def test_bitwise_or(self):
        """Tests that bitwise OR combines relations correctly."""
        combined = IntervalRelation.PRECEDES | IntervalRelation.MEETS
        assert isinstance(combined, IntervalRelation)
        assert IntervalRelation.PRECEDES in combined
        assert IntervalRelation.MEETS in combined
        assert IntervalRelation.EQUAL not in combined

    def test_bitwise_and(self):
        """Tests that bitwise AND intersects relations correctly."""
        combined = IntervalRelation.PRECEDES | IntervalRelation.MEETS
        result = combined & IntervalRelation.PRECEDES
        assert isinstance(result, IntervalRelation)
        assert int(result) == int(IntervalRelation.PRECEDES)

    def test_matches_method(self):
        """Tests the matches method for checking relation membership."""
        mask = IntervalRelation.PRECEDES | IntervalRelation.MEETS
        assert IntervalRelation.PRECEDES.matches(mask)
        assert IntervalRelation.MEETS.matches(mask)
        assert not IntervalRelation.EQUAL.matches(mask)

    def test_composite_relations(self):
        """Tests the composite relation class methods."""
        before = IntervalRelation.BEFORE()
        assert IntervalRelation.PRECEDES in before
        assert IntervalRelation.MEETS in before

        after = IntervalRelation.AFTER()
        assert IntervalRelation.PRECEDED_BY in after
        assert IntervalRelation.MET_BY in after

        subset = IntervalRelation.SUBSET()
        assert IntervalRelation.STARTS in subset
        assert IntervalRelation.DURING in subset
        assert IntervalRelation.FINISHES in subset
        assert IntervalRelation.EQUAL in subset

        superset = IntervalRelation.SUPERSET()
        assert IntervalRelation.STARTED_BY in superset
        assert IntervalRelation.CONTAINS in superset
        assert IntervalRelation.FINISHED_BY in superset
        assert IntervalRelation.EQUAL in superset

    def test_shape_property(self):
        """Tests that shape property works correctly."""
        scalar_rel = IntervalRelation.EQUAL
        assert scalar_rel.shape == ()

        array_vals = jnp.array([1, 2, 4])
        array_rel = IntervalRelation(array_vals)
        assert array_rel.shape == (3,)

    def test_pytree_registration(self):
        """Tests that IntervalRelation works with JAX pytree operations."""
        rel = IntervalRelation.PRECEDES
        leaves, treedef = jax.tree_util.tree_flatten(rel)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(reconstructed, IntervalRelation)
        assert int(reconstructed) == int(rel)
