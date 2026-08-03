import inspect
import sys

import pytest

import numpy as np
from numpy import random
from numpy.testing import assert_, assert_array_equal, assert_raises


class TestRegression:

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = random.mtrand.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(np.random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(np.random.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # Check for 32-bit systems
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # Check for 64-bit systems
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(np.random.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        np.random.seed(0)
        rvsn = np.random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            rng = np.random.RandomState(12345)
            shuffled = list(t)
            rng.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = np.random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            np.random.seed(i)
            m.seed(4321)
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        np.random.multivariate_normal([0], [[0]], size=1)
        np.random.multivariate_normal([0], [[0]], size=np.int_(1))
        np.random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        np.random.seed(1234567890)
        x = np.random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in np.random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        np.random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = np.random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, np.random.choice, a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        np.random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            np.random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        np.random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            np.random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        rng = np.random.RandomState(1)
        orig = np.arange(3).view(N)
        perm = rng.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        rng = np.random.RandomState(1)
        m = M()
        perm = rng.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))

    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    @pytest.mark.parametrize(
        "cls",
        [
            random.Generator,
            random.MT19937,
            random.PCG64,
            random.PCG64DXSM,
            random.Philox,
            random.RandomState,
            random.SFC64,
            random.BitGenerator,
            random.SeedSequence,
            random.bit_generator.SeedlessSeedSequence,
        ],
    )
    def test_inspect_signature(self, cls: type) -> None:
        assert hasattr(cls, "__text_signature__")
        try:
            inspect.signature(cls)
        except ValueError:
            pytest.fail(f"invalid signature: {cls.__module__}.{cls.__qualname__}")


@pytest.mark.parametrize(
    "bitgen_cls",
    [
        random.PCG64,
        random.PCG64DXSM,
        random.MT19937,
        random.Philox,
        random.SFC64,
    ],
)
class TestBitGeneratorFailedReinit:
    """Regression tests for gh-28784.

    Failed BitGenerator.__init__ must not leave a previously valid instance
    with a NULL C state pointer (which used to make random_raw segfault).
    """

    def test_failed_reinit_preserves_usable_state(self, bitgen_cls):
        bg = bitgen_cls(12345)
        before = bg.random_raw(5)
        bg.state  # ensure state property works
        with pytest.raises(ValueError):
            bg.__init__(("",))
        # Prior state must remain usable (no segfault, no ValueError).
        after = bg.random_raw(5)
        assert_(isinstance(after, np.ndarray))
        assert_array_equal(after.shape, (5,))
        # Stream advanced from the successful draws before the failed reinit.
        assert_(not np.array_equal(before, after))

    def test_failed_reinit_preserves_state_dict(self, bitgen_cls):
        bg = bitgen_cls(999)
        state_before = bg.state
        raw_before = bg.random_raw()
        with pytest.raises((ValueError, TypeError)):
            bg.__init__([1.5])
        # Restore the pre-draw state and confirm the generator still works.
        bg.state = state_before
        assert_array_equal(bg.random_raw(), raw_before)

    def test_failed_reinit_negative_seed(self, bitgen_cls):
        bg = bitgen_cls(7)
        state_before = bg.state
        with pytest.raises(ValueError):
            bg.__init__(-1)
        bg.state = state_before
        assert_(isinstance(bg.random_raw(), (int, np.integer)))

    def test_issue_reproducer_invalid_seed_tuple(self, bitgen_cls):
        # Pattern from gh-28784 (invalid seed tuple of str).
        bg = bitgen_cls()
        with pytest.raises(ValueError):
            bg.__init__(("",))
        # Must not SIGSEGV.
        raw = bg.random_raw()
        assert_(isinstance(raw, (int, np.integer)))


class TestIncompleteBitGeneratorSubclass:
    """Incomplete BitGenerator subclasses must fail cleanly (related: gh-25023)."""

    def test_random_raw_uninitialized(self):
        class FakeRandom(random.BitGenerator):
            def __init__(self, seed=None):
                super().__init__(seed)

            def spawn(self, n):
                raise NotImplementedError

        bg = FakeRandom()
        with pytest.raises(ValueError, match="not fully initialized"):
            bg.random_raw()

    def test_generator_rejects_uninitialized(self):
        class FakeRandom(random.BitGenerator):
            def __init__(self, seed=None):
                super().__init__(seed)

        bg = FakeRandom()
        with pytest.raises(ValueError, match="not fully initialized"):
            random.Generator(bg)


class TestPhiloxFailedReinit:
    def test_invalid_counter_does_not_break_instance(self):
        bg = random.Philox(123)
        state = bg.state
        raw = bg.random_raw(3)
        with pytest.raises((ValueError, TypeError, OverflowError)):
            bg.__init__(seed=1, counter="not-an-int")
        bg.state = state
        assert_array_equal(bg.random_raw(3), raw)
