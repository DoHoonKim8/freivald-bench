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
//!
//! This uses `m + n` dot products of length `n`, and `n` dot products of length `m`.
//! If `n = m`, then we do `3n` dot products of length `n`.
//!
//! For a normal matrix multiplication, we need n^2 dot products of length `n`.

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

/// Matrix type
#[derive(Clone, Debug)]
pub struct Array<F: Field> {
    vals: Vec<Vec<Value<F>>>,
    num_cols: usize,
    num_rows: usize,
}

impl<F: Field> Array<F> {
    fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Random generate matrix
    pub fn rand<R: RngCore>(rng: &mut R, num_cols: usize, num_rows: usize) -> Self {
        let vals = vec![(); num_cols]
            .iter()
            .map(|_| {
                vec![(); num_rows]
                    .iter()
                    .map(|_| Value::known(F::random(&mut *rng)))
                    .collect()
            })
            .collect();
        Self {
            vals,
            num_cols,
            num_rows,
        }
    }

    fn transpose(&self) -> Self {
        let mut transpose = vec![vec![Value::unknown(); self.num_cols]; self.num_rows];
        for (col_idx, col) in self.vals.iter().enumerate() {
            for (row_idx, val) in col.iter().enumerate() {
                transpose[row_idx][col_idx] = *val;
            }
        }

        Self {
            vals: transpose,
            num_cols: self.num_rows,
            num_rows: self.num_cols,
        }
    }

    /// Matrix multiplication
    pub fn mat_mul(&self, matrix: &Self) -> Self {
        let vals = matrix
            .vals
            .iter()
            .map(|col| self.vec_mul(col))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            vals,
            num_cols: matrix.num_cols,
            num_rows: self.num_rows,
        }
    }

    /// A • v
    pub fn vec_mul(&self, vector: &[Value<F>]) -> Vec<Value<F>> {
        assert_eq!(vector.len(), self.num_cols);
        self.transpose()
            .vals
            .iter()
            .map(|row| dot_product(row, vector))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

fn dot_product<F: Field>(left: &[Value<F>], right: &[Value<F>]) -> Value<F> {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| *left * right)
        .fold(Value::known(F::ZERO), |acc, v| acc + v)
}

/// Freivald algorithm circuit configuration
#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct FreivaldConfig {
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
impl FreivaldConfig {
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

/// Freivald algorithm circuit
#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct FreivaldCircuit<F: Field> {
    // H x W
    A: Array<F>,
    // W x H
    B: Array<F>,
    // H x H
    C: Array<F>,
    H: usize,
    W: usize,
}

impl<F: Field> FreivaldCircuit<F> {
    /// Create Freivald algorithm circuit
    #[allow(non_snake_case)]
    pub fn new(A: Array<F>, B: Array<F>, C: Array<F>, H: usize, W: usize) -> Self {
        Self { A, B, C, H, W }
    }
}

#[allow(non_snake_case)]
impl<F: Field> Circuit<F> for FreivaldCircuit<F> {
    type Config = FreivaldConfig;
    type FloorPlanner = V1;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        FreivaldConfig::configure(meta)
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

                for (row_idx, row) in self.B.transpose().vals.iter().enumerate() {
                    let running_sum = row.iter().scan(Value::known(F::ZERO), |state, val| {
                        *state = (*state + val) * r;
                        Some(*state)
                    });

                    for (idx, (b, br)) in row.iter().zip(running_sum).enumerate() {
                        let offset = row_idx * self.B.num_cols() + idx;

                        region.assign_advice(|| "", config.B, offset, || *b)?;
                        let Br = region.assign_advice(|| "", config.Br, offset, || br)?;
                        if idx == self.H - 1 {
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

                for (row_idx, row) in self.C.transpose().vals.iter().enumerate() {
                    let running_sum = row.iter().scan(Value::known(F::ZERO), |state, val| {
                        *state = (*state + val) * r;
                        Some(*state)
                    });

                    for (idx, (c, cr)) in row.iter().zip(running_sum).enumerate() {
                        let offset = row_idx * self.C.num_cols() + idx;

                        region.assign_advice(|| "", config.C, offset, || *c)?;
                        let Cr = region.assign_advice(|| "", config.Cr, offset, || cr)?;
                        if idx == self.H - 1 {
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
                let mut Br_vals = vec![Value::unknown(); self.W];
                for (idx, Br_cell) in Br.iter().enumerate() {
                    Br_vals[idx] = Br_cell.value().copied();
                }

                let mut ABr_cells = vec![];
                for (row_idx, row) in self.A.transpose().vals.iter().enumerate() {
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

                        if idx == self.W - 1 {
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
