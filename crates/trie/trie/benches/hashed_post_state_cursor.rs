#![allow(missing_docs, unreachable_pub)]
#![recursion_limit = "256"]

use alloy_primitives::{B256, U256};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use reth_primitives_traits::Account;
use reth_trie::{
    hashed_cursor::{noop::NoopHashedCursor, HashedCursor, HashedPostStateCursor},
    HashedPostStateSorted,
};

const ENTRIES_PER_DATASET: usize = 100_000;

/// Generate N random sorted account datasets. Each dataset has `ENTRIES_PER_DATASET` entries.
fn generate_datasets(n: usize) -> Vec<Vec<(B256, Option<Account>)>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            let mut entries: Vec<(B256, Option<Account>)> = (0..ENTRIES_PER_DATASET)
                .map(|_| {
                    let key = B256::from(rng.random::<[u8; 32]>());
                    let account = Account {
                        nonce: rng.random::<u64>(),
                        balance: U256::from(rng.random::<u128>()),
                        bytecode_hash: Some(B256::from(rng.random::<[u8; 32]>())),
                    };
                    (key, Some(account))
                })
                .collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            entries.dedup_by_key(|e| e.0);
            entries
        })
        .collect()
}

/// Wrapper to make `HashedCursor` object-safe via dynamic dispatch.
struct BoxedHashedCursor(Box<dyn HashedCursor<Value = Account>>);

impl HashedCursor for BoxedHashedCursor {
    type Value = Account;

    fn seek(&mut self, key: B256) -> Result<Option<(B256, Account)>, reth_storage_errors::db::DatabaseError> {
        self.0.seek(key)
    }

    fn next(&mut self) -> Result<Option<(B256, Account)>, reth_storage_errors::db::DatabaseError> {
        self.0.next()
    }

    fn reset(&mut self) {
        self.0.reset()
    }
}

/// Flatten N datasets into a single sorted dataset, create one cursor, and iterate.
fn flatten_and_iterate(datasets: &[Vec<(B256, Option<Account>)>]) -> usize {
    let mut merged = Vec::with_capacity(datasets.iter().map(|d| d.len()).sum());
    for dataset in datasets {
        merged.extend_from_slice(dataset);
    }
    merged.sort_unstable_by_key(|(k, _)| *k);
    merged.dedup_by_key(|e| e.0);

    let sorted = HashedPostStateSorted::new(merged, Default::default());
    let base: NoopHashedCursor<Account> = NoopHashedCursor::default();
    let mut cursor = HashedPostStateCursor::new_account(base, &sorted);

    let mut count = 0usize;
    cursor.seek(B256::ZERO).unwrap();
    count += 1;
    while cursor.next().unwrap().is_some() {
        count += 1;
    }
    count
}

/// Stack N `HashedPostStateCursor`s over a noop base and iterate the full merged view.
///
/// Uses `BoxedHashedCursor` to erase the recursive generic type at each layer,
/// since N varies at runtime and the compiler can't monomorphize 128-deep nesting.
fn stack_and_iterate(sorted_states: &[HashedPostStateSorted]) -> usize {
    let mut cursor: BoxedHashedCursor =
        BoxedHashedCursor(Box::new(NoopHashedCursor::<Account>::default()));

    for state in sorted_states {
        // SAFETY: we transmute the lifetime of `state` to 'static so it can be stored
        // inside the Box. This is safe because `sorted_states` outlives `cursor` — we
        // consume and drop `cursor` before this function returns.
        let state_ref: &'static HashedPostStateSorted =
            unsafe { std::mem::transmute::<&HashedPostStateSorted, &'static HashedPostStateSorted>(state) };
        let inner = HashedPostStateCursor::new_account(cursor, state_ref);
        cursor = BoxedHashedCursor(Box::new(inner));
    }

    let mut count = 0usize;
    cursor.seek(B256::ZERO).unwrap();
    count += 1;
    while cursor.next().unwrap().is_some() {
        count += 1;
    }
    count
}

fn bench_cursor_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashedPostStateCursor");
    group.sample_size(10);

    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
        let datasets = generate_datasets(n);

        let sorted_states: Vec<HashedPostStateSorted> = datasets
            .iter()
            .map(|d| HashedPostStateSorted::new(d.clone(), Default::default()))
            .collect();

        group.bench_function(BenchmarkId::new("flattened", n), |b| {
            b.iter(|| flatten_and_iterate(&datasets));
        });

        group.bench_function(BenchmarkId::new("stacked", n), |b| {
            b.iter(|| stack_and_iterate(&sorted_states));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cursor_iteration);
criterion_main!(benches);
