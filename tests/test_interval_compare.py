import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.comparison import IntervalRelation, interval_compare

# =============================================================================
# Interval Fixtures: Shape (1,)
# =============================================================================


@pytest.fixture
def interval_1d_1_2():
    """Interval [1, 2] with shape (1,)."""
    return irx.interval(jnp.array([1.0]), jnp.array([2.0]))


@pytest.fixture
def interval_1d_1_3():
    """Interval [1, 3] with shape (1,)."""
    return irx.interval(jnp.array([1.0]), jnp.array([3.0]))


@pytest.fixture
def interval_1d_1_4():
    """Interval [1, 4] with shape (1,)."""
    return irx.interval(jnp.array([1.0]), jnp.array([4.0]))


@pytest.fixture
def interval_1d_2_3():
    """Interval [2, 3] with shape (1,)."""
    return irx.interval(jnp.array([2.0]), jnp.array([3.0]))


@pytest.fixture
def interval_1d_2_4():
    """Interval [2, 4] with shape (1,)."""
    return irx.interval(jnp.array([2.0]), jnp.array([4.0]))


@pytest.fixture
def interval_1d_3_4():
    """Interval [3, 4] with shape (1,)."""
    return irx.interval(jnp.array([3.0]), jnp.array([4.0]))


@pytest.fixture
def interval_1d_3_5():
    """Interval [3, 5] with shape (1,)."""
    return irx.interval(jnp.array([3.0]), jnp.array([5.0]))


# =============================================================================
# Interval Fixtures: Singletons (shape (1,), lower == upper)
# =============================================================================


@pytest.fixture
def interval_1d_2_2():
    """Singleton interval [2, 2] with shape (1,)."""
    return irx.interval(jnp.array([2.0]), jnp.array([2.0]))


@pytest.fixture
def interval_1d_2p5_2p5():
    """Singleton interval [2.5, 2.5] with shape (1,)."""
    return irx.interval(jnp.array([2.5]), jnp.array([2.5]))


@pytest.fixture
def interval_1d_3_3():
    """Singleton interval [3, 3] with shape (1,)."""
    return irx.interval(jnp.array([3.0]), jnp.array([3.0]))


# =============================================================================
# Interval Fixtures: Empty (shape (1,), lower > upper)
# =============================================================================


@pytest.fixture
def interval_1d_empty_3_1():
    """Empty interval [3, 1] with shape (1,) (lower > upper)."""
    return irx.interval(jnp.array([3.0]), jnp.array([1.0]))


@pytest.fixture
def interval_1d_empty_5_2():
    """Empty interval [5, 2] with shape (1,) (lower > upper)."""
    return irx.interval(jnp.array([5.0]), jnp.array([2.0]))


# =============================================================================
# Interval Fixtures: Scalar (shape ())
# =============================================================================


@pytest.fixture
def interval_scalar_1_2():
    """Scalar interval [1, 2] with shape ()."""
    return irx.interval(jnp.array(1.0), jnp.array(2.0))


@pytest.fixture
def interval_scalar_5_6():
    """Scalar interval [5, 6] with shape ()."""
    return irx.interval(jnp.array(5.0), jnp.array(6.0))


# =============================================================================
# Interval Fixtures: Shape (3,)
# =============================================================================


@pytest.fixture
def interval_3d_123_234():
    """Interval [[1,2], [2,3], [3,4]] with shape (3,)."""
    return irx.interval(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 3.0, 4.0]))


@pytest.fixture
def interval_3d_567_678():
    """Interval [[5,6], [6,7], [7,8]] with shape (3,)."""
    return irx.interval(jnp.array([5.0, 6.0, 7.0]), jnp.array([6.0, 7.0, 8.0]))


# =============================================================================
# Interval Fixtures: Shape (4,)
# =============================================================================


@pytest.fixture
def interval_4d_1234_2345():
    """Interval [[1,2], [2,3], [3,4], [4,5]] with shape (4,)."""
    return irx.interval(
        jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([2.0, 3.0, 4.0, 5.0])
    )


# =============================================================================
# Parametrized Relation Fixtures
# =============================================================================

# Standard Allen relations - tuples of (fixture_name_a, fixture_name_b, expected_relation)
allen_relation_params = [
    pytest.param(
        ("interval_1d_1_2", "interval_1d_3_4", IntervalRelation.PRECEDES),
        id="PRECEDES",
    ),
    pytest.param(
        ("interval_1d_1_2", "interval_1d_2_3", IntervalRelation.MEETS),
        id="MEETS",
    ),
    pytest.param(
        ("interval_1d_1_3", "interval_1d_2_4", IntervalRelation.OVERLAPS),
        id="OVERLAPS",
    ),
    pytest.param(
        ("interval_1d_1_2", "interval_1d_1_3", IntervalRelation.STARTS),
        id="STARTS",
    ),
    pytest.param(
        ("interval_1d_2_3", "interval_1d_1_4", IntervalRelation.DURING),
        id="DURING",
    ),
    pytest.param(
        ("interval_1d_2_3", "interval_1d_1_3", IntervalRelation.FINISHES),
        id="FINISHES",
    ),
    pytest.param(
        ("interval_1d_1_2", "interval_1d_1_2", IntervalRelation.EQUAL),
        id="EQUAL",
    ),
]


@pytest.fixture(params=allen_relation_params)
def allen_relation_pair(request):
    """Parametrized fixture returning (interval_a, interval_b, expected_relation)."""
    fixture_a, fixture_b, expected = request.param
    a = request.getfixturevalue(fixture_a)
    b = request.getfixturevalue(fixture_b)
    return (a, b, expected)


# Edge case relations - singleton and boundary intervals
edge_case_params = [
    pytest.param(
        ("interval_1d_2_2", "interval_1d_3_3", IntervalRelation.PRECEDES),
        id="SINGLETON_PRECEDES_SINGLETON",
    ),
    pytest.param(
        ("interval_1d_2_2", "interval_1d_2_2", IntervalRelation.EQUAL),
        id="SINGLETON_EQUAL_SINGLETON",
    ),
    pytest.param(
        ("interval_1d_2_2", "interval_1d_2_3", IntervalRelation.STARTS),
        id="SINGLETON_STARTS_NORMAL",
    ),
    pytest.param(
        ("interval_1d_3_3", "interval_1d_2_3", IntervalRelation.FINISHES),
        id="SINGLETON_FINISHES_NORMAL",
    ),
    pytest.param(
        ("interval_1d_2p5_2p5", "interval_1d_2_3", IntervalRelation.DURING),
        id="SINGLETON_DURING_NORMAL",
    ),
    pytest.param(
        ("interval_1d_3_5", "interval_1d_1_3", IntervalRelation.MET_BY),
        id="NORMAL_MET_BY_NORMAL",
    ),
]


@pytest.fixture(params=edge_case_params)
def edge_case_pair(request):
    """Parametrized fixture for edge case interval pairs."""
    fixture_a, fixture_b, expected = request.param
    a = request.getfixturevalue(fixture_a)
    b = request.getfixturevalue(fixture_b)
    return (a, b, expected)


# Broadcasting test cases
broadcasting_params = [
    pytest.param(
        ("interval_scalar_1_2", "interval_scalar_5_6", IntervalRelation.PRECEDES, ()),
        id="SCALAR_PRECEDES_SCALAR",
    ),
    pytest.param(
        ("interval_scalar_1_2", "interval_3d_567_678", IntervalRelation.PRECEDES, (3,)),
        id="SCALAR_BROADCASTS_TO_3D",
    ),
    pytest.param(
        ("interval_3d_123_234", "interval_3d_567_678", IntervalRelation.PRECEDES, (3,)),
        id="3D_PRECEDES_3D",
    ),
]


@pytest.fixture(params=broadcasting_params)
def broadcasting_pair(request):
    """Parametrized fixture for broadcasting test cases."""
    fixture_a, fixture_b, expected, expected_shape = request.param
    a = request.getfixturevalue(fixture_a)
    b = request.getfixturevalue(fixture_b)
    return (a, b, expected, expected_shape)


# =============================================================================
# Test Functions: Standard Allen Relations
# =============================================================================


def test_interval_compare_allen_relations(allen_relation_pair):
    """Tests interval_compare for all 7 unique Allen relations.

    Verifies:
    1. Return type is IntervalRelation
    2. Result shape matches input shapes
    3. Result equals the expected relation
    """
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
# Test Functions: Edge Cases
# =============================================================================


def test_interval_compare_edge_cases(edge_case_pair):
    """Tests interval_compare for edge cases (singleton and boundary intervals).

    Verifies:
    1. Return type is IntervalRelation
    2. Result shape matches input shapes
    3. Result equals the expected relation
    """
    a, b, expected = edge_case_pair
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
    """Tests interval_compare broadcasting behavior.

    Verifies:
    1. Return type is IntervalRelation
    2. Result shape matches expected broadcasted shape
    3. Result equals the expected relation
    """
    a, b, expected, expected_shape = broadcasting_pair
    result = interval_compare(a, b)

    assert isinstance(result, IntervalRelation), (
        f"Expected IntervalRelation, got {type(result).__name__}"
    )
    assert result.shape == expected_shape, (
        f"Result shape {result.shape} does not match expected shape {expected_shape}"
    )
    assert jnp.all(result == expected), f"Expected {expected}, got {result}"


def test_interval_compare_incompatible_shapes_raises(
    interval_3d_123_234, interval_4d_1234_2345
):
    """Tests that comparing intervals with incompatible shapes raises an error."""
    with pytest.raises(Exception):  # JAX raises ValueError for shape mismatch
        interval_compare(interval_3d_123_234, interval_4d_1234_2345)


def test_interval_compare_broadcasting_matches_elementwise(
    interval_3d_123_234, interval_3d_567_678
):
    """Tests that JAX broadcasting gives same result as manual elementwise comparison.

    Compares each pair of scalar intervals individually and concatenates results,
    then verifies this matches the broadcasted interval_compare result.
    """
    a = interval_3d_123_234
    b = interval_3d_567_678

    # Broadcasted comparison
    broadcasted_result = interval_compare(a, b)

    # Manual elementwise comparison
    elementwise_results = []
    for i in range(3):
        a_i = irx.interval(jnp.array([a.lower[i]]), jnp.array([a.upper[i]]))
        b_i = irx.interval(jnp.array([b.lower[i]]), jnp.array([b.upper[i]]))
        rel = interval_compare(a_i, b_i)
        elementwise_results.append(rel.value[0])

    elementwise_array = jnp.array(elementwise_results)
    elementwise_result = IntervalRelation(elementwise_array)

    assert jnp.all(broadcasted_result.value == elementwise_result.value), (
        f"Broadcasted result {broadcasted_result.value} does not match "
        f"elementwise result {elementwise_result.value}"
    )


def test_interval_compare_broadcasting_mixed_relations():
    """Tests broadcasting with intervals that have different relations per element."""
    # Element 0: [1,2] PRECEDES [5,6]
    # Element 1: [2,4] OVERLAPS [3,5]
    # Element 2: [1,3] EQUAL [1,3]
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
        f"Expected relations {expected_values}, got {result.value}"
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
