//! Dot product

use criterion::{criterion_group, criterion_main, Criterion};
use halo2_proofs::halo2curves::bn256::{Bn256, Fr};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{floor_planner::V1, Layouter, Value},
    plonk::*,
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use rand::{rngs::OsRng, RngCore};

#[derive(Clone, Copy)]
struct MyConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    ab: Column<Advice>,
    q_init: Selector,
    q_running_sum: Selector,
}

#[allow(non_snake_case)]
impl MyConfig {
    fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let ab = meta.advice_column();

        let q_init = meta.selector();
        let q_running_sum = meta.selector();

        meta.create_gate("ab[0] = a[0] • b[0]", |_| {
            let q_init = q_init.expr();
            let a = a.cur();
            let b = b.cur();
            let ab = ab.cur();

            vec![q_init * (a * b - ab)]
        });

        // ∑ a[i] • b[i]
        meta.create_gate("ab[i] = ab[i-1] + a[i] * b[i]", |_| {
            let q_running_sum = q_running_sum.expr();
            let a = a.cur();
            let b = b.cur();
            let ab_cur = ab.cur();
            let ab_prev = ab.prev();

            vec![q_running_sum * (ab_cur - (ab_prev + a * b))]
        });

        Self {
            a,
            b,
            ab,
            q_init,
            q_running_sum,
        }
    }

    fn assign<F: Field>(
        &self,
        a: &[Value<F>],
        b: &[Value<F>],
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "A • Br",
            |mut region| {
                let ab = a
                    .iter()
                    .zip(b.iter())
                    .scan(Value::known(F::ZERO), |state, (a, b)| {
                        *state = *state + *a * b;
                        Some(*state)
                    });

                for (idx, ((a, b), ab)) in a.iter().zip(b.iter()).zip(ab).enumerate() {
                    region.assign_advice(|| "", self.a, idx, || *a)?;
                    region.assign_advice(|| "", self.b, idx, || *b)?;
                    region.assign_advice(|| "", self.ab, idx, || ab)?;

                    if idx == 0 {
                        self.q_init.enable(&mut region, idx)?;
                    } else {
                        self.q_running_sum.enable(&mut region, idx)?;
                    }
                }

                Ok(())
            },
        )
    }
}

#[derive(Clone)]
#[allow(non_snake_case)]
struct MyCircuit<F: Field> {
    // Vector of (a, b)'s
    values: Vec<(Vec<Value<F>>, Vec<Value<F>>)>,
}

#[allow(non_snake_case)]
impl<F: Field> Circuit<F> for MyCircuit<F> {
    // This is to match the number of advice columns in Freivalds; here,
    // we have 4 * 2 = 8 advice columns; while Freivalds has 7 advice columns
    type Config = [MyConfig; 4];
    type FloorPlanner = V1;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        [
            MyConfig::configure(meta),
            MyConfig::configure(meta),
            MyConfig::configure(meta),
            MyConfig::configure(meta),
        ]
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chunk_size = (self.values.len() as u32).div_ceil(4) as usize;
        for (chunk_idx, chunk) in self.values.chunks(chunk_size).enumerate() {
            let config = config[chunk_idx];
            for (a, b) in chunk.iter() {
                config.assign(a, b, layouter.namespace(|| ""))?;
            }
        }

        Ok(())
    }
}

fn bench(name: &str, num_dot_products: usize, n: usize, k: u32, c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    group.sample_size(10);
    // Initialize the polynomial commitment parameters
    let params: ParamsKZG<Bn256> = ParamsKZG::new(k);

    let values = (0..num_dot_products)
        .map(|_| {
            let a = (0..n)
                .map(|_| Value::known(Fr::random(&mut OsRng)))
                .collect();
            let b = (0..n)
                .map(|_| Value::known(Fr::random(&mut OsRng)))
                .collect();
            (a, b)
        })
        .collect();

    let circuit = &MyCircuit { values };
    let vk = keygen_vk(&params, circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, circuit).expect("keygen_pk should not fail");

    let prover_name = name.to_string() + "-prover";
    group.bench_function(&prover_name, |b| {
        b.iter(|| {
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

            create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
                &params,
                &pk,
                &[circuit.clone()],
                &[&[]],
                OsRng,
                &mut transcript,
            )
            .expect("proof generation should not fail");

            transcript.finalize();
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench(
        "num_dot_products = 10000, N = 100, K = 18",
        10000,
        100,
        18,
        c,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
