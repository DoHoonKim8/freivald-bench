//! Dot product

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

/// Dot product circuit configuration
#[derive(Clone, Copy, Debug)]
pub struct DotProductConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    ab: Column<Advice>,
    q_init: Selector,
    q_running_sum: Selector,
}

#[allow(non_snake_case)]
impl DotProductConfig {
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

/// Dot product circuit
#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct DotProductCircuit<F: Field> {
    // Vector of (a, b)'s
    values: Vec<(Vec<Value<F>>, Vec<Value<F>>)>,
}

impl<F: Field> DotProductCircuit<F> {
    /// Create dot product circuit
    pub fn new(values: Vec<(Vec<Value<F>>, Vec<Value<F>>)>) -> Self {
        Self { values }
    }
}

#[allow(non_snake_case)]
impl<F: Field> Circuit<F> for DotProductCircuit<F> {
    // This is to match the number of advice columns in Freivalds; here,
    // we have 3 * 3 = 9 advice columns; while Freivalds has 7 advice columns
    type Config = [DotProductConfig; 3];
    type FloorPlanner = V1;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        [
            DotProductConfig::configure(meta),
            DotProductConfig::configure(meta),
            DotProductConfig::configure(meta),
        ]
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chunk_size = (self.values.len() as u32).div_ceil(3) as usize;
        for (chunk_idx, chunk) in self.values.chunks(chunk_size).enumerate() {
            let config = config[chunk_idx];
            for (a, b) in chunk.iter() {
                config.assign(a, b, layouter.namespace(|| ""))?;
            }
        }

        Ok(())
    }
}
