//! Freivalds' algorithm to verify matrix multiplication
//!
//! To verify A • B = C, where:
//!     - A is an n x m matrix,
//!     - B is an m x n matrix, and
//!     - C is an n x n matrix
//!
//! 1. Sample random vector r <- [F; n]
//! 2. Define b = B r,
//!           c = C r
//! 3. Constrain A b = c

use ff::FromUniformBytes;
use halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    circuit::{floor_planner::V1, Layouter, Value},
    dev::{metadata::Constraint, FailureLocation, MockProver, VerifyFailure},
    plonk::*,
    poly::{
        commitment::{Params, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{EpAffine, Fq};
use rand_core::{OsRng, RngCore};

#[derive(Clone, Copy, Debug)]
struct Array<F: Field, const W: usize, const H: usize>([[Value<F>; H]; W]);

impl<F: Field, const W: usize, const H: usize> Array<F, W, H> {
    fn num_cols(&self) -> usize {
        W
    }

    fn rand<R: RngCore>(rng: &mut R) -> Self {
        Self([(); W].map(|_| [(); H].map(|_| Value::known(F::random(&mut *rng)))))
    }

    fn transpose(&self) -> Array<F, H, W> {
        let mut transpose = [[Value::unknown(); W]; H];
        for (col_idx, col) in self.0.iter().enumerate() {
            for (row_idx, val) in col.iter().enumerate() {
                transpose[row_idx][col_idx] = *val;
            }
        }
        Array(transpose)
    }

    fn mat_mul(&self, matrix: Array<F, H, W>) -> Array<F, H, H> {
        Array(
            matrix
                .0
                .iter()
                .map(|col| self.vec_mul(*col))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    /// A • v
    fn vec_mul(&self, vector: [Value<F>; W]) -> [Value<F>; H] {
        self.transpose()
            .0
            .iter()
            .map(|row| dot_product(*row, vector))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

fn dot_product<F: Field, const L: usize>(left: [Value<F>; L], right: [Value<F>; L]) -> Value<F> {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| *left * right)
        .fold(Value::known(F::ZERO), |acc, v| acc + v)
}

#[derive(Clone)]
#[allow(non_snake_case)]
struct MyConfig {
    /// Vertically stack the rows of A
    /// Each row has W elements; there are H rows
    A: Column<Advice>,
    /// Vertically stack the rows of B
    /// Each row has H elements; there are W rows
    B: Column<Advice>,
    /// Vertically stack the rows of C
    /// Each row has H elements; there are H rows
    C: Column<Advice>,
    /// Matrix-vector mul B • r
    Br: Column<Advice>,
    q_Br: (Selector, Selector),
    /// Matrix-vector mul C • r
    Cr: Column<Advice>,
    q_Cr: (Selector, Selector),
    /// Matrix-vector mul A • (B • r)
    ABr: (Column<Advice>, Column<Advice>),
    q_ABr: (Selector, Selector),
    r: Challenge,
}

#[allow(non_snake_case)]
impl MyConfig {
    fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
        let q_Br = (meta.selector(), meta.selector());
        let q_Cr = (meta.selector(), meta.selector());
        let q_ABr = (meta.selector(), meta.selector());

        // First phase
        let A = meta.advice_column_in(FirstPhase);
        let B = meta.advice_column_in(FirstPhase);
        let C = meta.advice_column_in(FirstPhase);

        let r = meta.challenge_usable_after(FirstPhase);

        // Second phase
        let Br = meta.advice_column_in(SecondPhase);
        let Cr = meta.advice_column_in(SecondPhase);
        let ABr = (
            meta.advice_column_in(SecondPhase),
            meta.advice_column_in(SecondPhase),
        );

        meta.enable_equality(Br);
        meta.enable_equality(Cr);
        meta.enable_equality(ABr.0);
        meta.enable_equality(ABr.1);

        meta.create_gate("Br[0] = B[0] • r", |_| {
            let q_Br_init = q_Br.0.expr();
            let B = B.cur();
            let Br = Br.cur();
            let r = r.expr();

            vec![q_Br_init * (Br - B * r)]
        });

        // ∑ B[i] • r^{n - i}
        meta.create_gate("Br[i] = r * (Br[i - 1] + B[i])", |_| {
            let q_Br = q_Br.1.expr();
            let B = B.cur();
            let Br_prev = Br.prev();
            let Br = Br.cur();
            let r = r.expr();

            vec![q_Br * (Br - r * (B + Br_prev))]
        });

        // ∑ C[i] • r^{n - i}
        meta.create_gate("Cr[0] = C[0] • r", |_| {
            let q_Cr_init = q_Cr.0.expr();
            let C = C.cur();
            let Cr = Cr.cur();
            let r = r.expr();

            vec![q_Cr_init * (Cr - C * r)]
        });

        meta.create_gate("Cr[i] = r * (Cr[i - 1] + C[i])", |_| {
            let q_Cr = q_Cr.1.expr();
            let C = C.cur();
            let Cr_prev = Cr.prev();
            let Cr = Cr.cur();
            let r = r.expr();

            vec![q_Cr * (Cr - r * (C + Cr_prev))]
        });

        meta.create_gate("ABr[0] = A[0] • Br[0]", |_| {
            let q_ABr_init = q_ABr.0.expr();
            let A = A.cur();
            let Br = ABr.0.cur();
            let ABr = ABr.1.cur();

            vec![q_ABr_init * (ABr - A * Br)]
        });

        // ∑ A[i] • Br[i]
        meta.create_gate("ABr[i] = ABr[i-1] + A[i] * Br[i]", |_| {
            let q_ABr = q_ABr.1.expr();
            let A = A.cur();
            let Br = ABr.0.cur();
            let ABr_prev = ABr.1.prev();
            let ABr = ABr.1.cur();

            vec![q_ABr * (ABr - (ABr_prev + A * Br))]
        });

        Self {
            A,
            B,
            C,
            Br,
            q_Br,
            Cr,
            q_Cr,
            ABr,
            q_ABr,
            r,
        }
    }
}

#[derive(Clone)]
#[allow(non_snake_case)]
struct MyCircuit<F: Field, const W: usize, const H: usize> {
    A: Array<F, W, H>,
    B: Array<F, H, W>,
    C: Array<F, H, H>,
}

#[allow(non_snake_case)]
impl<F: Field, const W: usize, const H: usize> Circuit<F> for MyCircuit<F, W, H> {
    type Config = MyConfig;
    type FloorPlanner = V1;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        MyConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let r = layouter.get_challenge(config.r);

        let Br = layouter.assign_region(
            || "B • r",
            |mut region| {
                let mut Br_cells = vec![];

                for (row_idx, row) in self.B.transpose().0.iter().enumerate() {
                    let running_sum = row.iter().scan(Value::known(F::ZERO), |state, val| {
                        *state = (*state + val) * r;
                        Some(*state)
                    });

                    for (idx, (b, br)) in row.iter().zip(running_sum).enumerate() {
                        let offset = row_idx * self.B.num_cols() + idx;

                        region.assign_advice(|| "", config.B, offset, || *b)?;
                        let Br = region.assign_advice(|| "", config.Br, offset, || br)?;
                        if idx == H - 1 {
                            Br_cells.push(Br);
                        }

                        if idx == 0 {
                            config.q_Br.0.enable(&mut region, offset)?;
                        } else {
                            config.q_Br.1.enable(&mut region, offset)?;
                        }
                    }
                }

                Ok(Br_cells)
            },
        )?;

        let Cr = layouter.assign_region(
            || "C • r",
            |mut region| {
                let mut Cr_cells = vec![];

                for (row_idx, row) in self.C.transpose().0.iter().enumerate() {
                    let running_sum = row.iter().scan(Value::known(F::ZERO), |state, val| {
                        *state = (*state + val) * r;
                        Some(*state)
                    });

                    for (idx, (c, cr)) in row.iter().zip(running_sum).enumerate() {
                        let offset = row_idx * self.C.num_cols() + idx;

                        region.assign_advice(|| "", config.C, offset, || *c)?;
                        let Cr = region.assign_advice(|| "", config.Cr, offset, || cr)?;
                        if idx == H - 1 {
                            Cr_cells.push(Cr);
                        }

                        if idx == 0 {
                            config.q_Cr.0.enable(&mut region, offset)?;
                        } else {
                            config.q_Cr.1.enable(&mut region, offset)?;
                        }
                    }
                }

                Ok(Cr_cells)
            },
        )?;

        let ABr = layouter.assign_region(
            || "A • Br",
            |mut region| {
                let mut Br_vals = [Value::unknown(); W];
                for (idx, Br_cell) in Br.iter().enumerate() {
                    Br_vals[idx] = Br_cell.value().copied();
                }

                let mut ABr_cells = vec![];
                for (row_idx, row) in self.A.transpose().0.iter().enumerate() {
                    let running_sum = row.iter().zip(Br_vals.iter()).scan(
                        Value::known(F::ZERO),
                        |state, (a, br)| {
                            *state = *state + *a * br;
                            Some(*state)
                        },
                    );

                    for (idx, ((a, br), abr)) in
                        row.iter().zip(Br.iter()).zip(running_sum).enumerate()
                    {
                        let offset = row_idx * self.A.num_cols() + idx;
                        region.assign_advice(|| "", config.A, offset, || *a)?;
                        br.copy_advice(|| "", &mut region, config.ABr.0, offset)?;
                        let ABr = region.assign_advice(|| "", config.ABr.1, offset, || abr)?;

                        if idx == 0 {
                            config.q_ABr.0.enable(&mut region, offset)?;
                        } else {
                            config.q_ABr.1.enable(&mut region, offset)?;
                        }

                        if idx == W - 1 {
                            ABr_cells.push(ABr);
                        }
                    }
                }

                Ok(ABr_cells)
            },
        )?;

        layouter.assign_region(
            || "",
            |mut region| {
                for (Cr, ABr) in Cr.iter().zip(ABr.iter()) {
                    region.constrain_equal(Cr.cell(), ABr.cell())?;
                }

                Ok(())
            },
        )?;

        Ok(())
    }
}

fn test_mock_prover<F: Ord + FromUniformBytes<64>, const W: usize, const H: usize>(
    k: u32,
    circuit: MyCircuit<F, W, H>,
    expected: Result<(), Vec<(Constraint, FailureLocation)>>,
) {
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    prover.assert_satisfied();
    match (prover.verify(), expected) {
        (Ok(_), Ok(_)) => {}
        (Err(err), Err(expected)) => {
            assert_eq!(
                err.into_iter()
                    .map(|failure| match failure {
                        VerifyFailure::ConstraintNotSatisfied {
                            constraint,
                            location,
                            ..
                        } => (constraint, location),
                        _ => panic!("MockProver::verify has result unmatching expected"),
                    })
                    .collect::<Vec<_>>(),
                expected
            )
        }
        (_, _) => panic!("MockProver::verify has result unmatching expected"),
    };
}

fn test_prover<C: CurveAffine, const W: usize, const H: usize>(
    k: u32,
    circuit: MyCircuit<C::Scalar, W, H>,
    expected: bool,
) where
    C::Scalar: FromUniformBytes<64>,
{
    let params = ParamsIPA::<C>::new(k);
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();

    let proof = {
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

        create_proof::<IPACommitmentScheme<C>, ProverIPA<C>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation should not fail");

        transcript.finalize()
    };

    let accepted = {
        let strategy = AccumulatorStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

        verify_proof::<IPACommitmentScheme<C>, VerifierIPA<C>, _, _, _>(
            &params,
            pk.get_vk(),
            strategy,
            &[&[]],
            &mut transcript,
            params.n(),
        )
        .map(|strategy| strategy.finalize())
        .unwrap_or_default()
    };

    assert_eq!(accepted, expected);
}

#[allow(non_snake_case)]
fn main() {
    const W: usize = 4;
    const H: usize = 4;
    const K: u32 = 8;

    let A = Array::<Fq, W, H>::rand(&mut OsRng);
    let B = Array::<Fq, H, W>::rand(&mut OsRng);
    let C = A.mat_mul(B);

    // Sanity check in plaintext
    {
        let r: [Value<Fq>; H] = (0..H)
            .map(|_| Value::known(Fq::random(OsRng)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let b = B.vec_mul(r);
        let c = C.vec_mul(r);
        let Ab = A.vec_mul(b);

        Ab.iter()
            .zip(c.iter())
            .for_each(|(Ab, c)| (Ab - c).assert_if_known(|v| *v == Fq::ZERO));
    }

    let circuit = &MyCircuit::<_, W, H> { A, B, C };

    {
        test_mock_prover(K, circuit.clone(), Ok(()));
        test_prover::<EpAffine, W, H>(K, circuit.clone(), true);
    }

    let wrong_circuit = MyCircuit::<_, W, H> {
        A,
        B,
        C: Array::rand(&mut OsRng),
    };
    {
        test_prover::<EpAffine, W, H>(K, wrong_circuit.clone(), false);
    }
}
