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
use itertools::Itertools;
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
pub struct FreivaldConfig<const NUM_INNER_COLS_A: usize, const NUM_INNER_COLS_B: usize> {
    /// Each row has W elements; there are H rows
    A: [Column<Advice>; NUM_INNER_COLS_A],
    /// Each row has H elements; there are W rows
    B: [Column<Advice>; NUM_INNER_COLS_B],
    /// Each row has H elements; there are H rows
    C: [Column<Advice>; NUM_INNER_COLS_B],
    /// Matrix-vector mul B • r
    Br: Column<Advice>,
    q_Br: (Selector, Selector),
    /// Matrix-vector mul C • r
    Cr: Column<Advice>,
    q_Cr: (Selector, Selector),
    /// Matrix-vector mul A • (B • r)
    ABr: ([Column<Advice>; NUM_INNER_COLS_A], Column<Advice>),
    q_ABr: (Selector, Selector),
    r: Challenge,
}

#[allow(non_snake_case)]
impl<const NUM_INNER_COLS_A: usize, const NUM_INNER_COLS_B: usize>
    FreivaldConfig<NUM_INNER_COLS_A, NUM_INNER_COLS_B>
{
    fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
        let q_Br = (meta.selector(), meta.selector());
        let q_Cr = (meta.selector(), meta.selector());
        let q_ABr = (meta.selector(), meta.selector());

        // First phase
        let A = [(); NUM_INNER_COLS_A].map(|_| meta.advice_column_in(FirstPhase));
        let B = [(); NUM_INNER_COLS_B].map(|_| meta.advice_column_in(FirstPhase));
        let C = [(); NUM_INNER_COLS_B].map(|_| meta.advice_column_in(FirstPhase));

        let r = meta.challenge_usable_after(FirstPhase);

        // Second phase
        let Br = meta.advice_column_in(SecondPhase);
        let Cr = meta.advice_column_in(SecondPhase);
        let ABr = (
            [(); NUM_INNER_COLS_A].map(|_| meta.advice_column_in(SecondPhase)),
            meta.advice_column_in(SecondPhase),
        );

        meta.enable_equality(Br);
        meta.enable_equality(Cr);
        ABr.0.iter().for_each(|col| meta.enable_equality(*col));
        meta.enable_equality(ABr.1);

        let powers_of_r = (0..NUM_INNER_COLS_B)
            .scan(Expression::Constant(F::ONE), |r_power, _| {
                *r_power = r_power.clone() * r.expr();
                Some(r_power.clone())
            })
            .collect_vec()
            .into_iter()
            .rev()
            .collect_vec();
        let r_powered_to_num_inner_cols = powers_of_r[0].clone() * r.expr();

        meta.create_gate(
            "Br[0] = ∑_{i=0}^{NUM_INNER_COLS - 1} B[0][i] * r^{NUM_INNER_COLS - i - 1}",
            |_| {
                let q_Br_init = q_Br.0.expr();
                let B_times_r = B
                    .iter()
                    .zip(powers_of_r.iter())
                    .map(|(b, r_power)| b.cur() * r_power.clone())
                    .sum();
                let Br = Br.cur();

                vec![q_Br_init * (Br - B_times_r)]
            },
        );

        meta.create_gate("Br[j] = r^{NUM_INNER_COLS} * Br[j - 1] + ∑ ...", |_| {
            let q_Br = q_Br.1.expr();
            let B_times_r = B
                .iter()
                .zip(powers_of_r.iter())
                .map(|(b, r_power)| b.cur() * r_power.clone())
                .sum();
            let Br_prev = Br.prev();
            let Br = Br.cur();

            vec![q_Br * (Br - (Br_prev * r_powered_to_num_inner_cols.clone() + B_times_r))]
        });

        meta.create_gate(
            "Cr[0] = ∑_{i=0}^{NUM_INNER_COLS - 1} C[0][i] * r^{NUM_INNER_COLS - i - 1}",
            |_| {
                let q_Cr_init = q_Cr.0.expr();
                let C_times_r = C
                    .iter()
                    .zip(powers_of_r.iter())
                    .map(|(c, r_power)| c.cur() * r_power.clone())
                    .sum();
                let Cr = Cr.cur();

                vec![q_Cr_init * (Cr - C_times_r)]
            },
        );

        meta.create_gate("Cr[j] = r^{NUM_INNER_COLS} * Cr[j - 1] + ∑ ...", |_| {
            let q_Cr = q_Cr.1.expr();
            let C_times_r = C
                .iter()
                .zip(powers_of_r.iter())
                .map(|(c, r_power)| c.cur() * r_power.clone())
                .sum();
            let Cr_prev = Cr.prev();
            let Cr = Cr.cur();

            vec![q_Cr * (Cr - (Cr_prev * r_powered_to_num_inner_cols + C_times_r))]
        });

        meta.create_gate(
            "ABr[0] = ∑_{i=0}^{NUM_INNER_COLS - 1} A[0][i] * Br[0][i]",
            |_| {
                let q_ABr_init = q_ABr.0.expr();
                let A_times_Br = A
                    .iter()
                    .zip(ABr.0.iter())
                    .map(|(a, br)| a.cur() * br.cur())
                    .sum();
                let ABr = ABr.1.cur();

                vec![q_ABr_init * (ABr - A_times_Br)]
            },
        );

        meta.create_gate(
            "ABr[j] = ABr[j-1] + ∑_{i=0}^{NUM_INNER_COLS - 1} A[j][i] * Br[j][i]",
            |_| {
                let q_ABr = q_ABr.1.expr();
                let A_times_Br = A
                    .iter()
                    .zip(ABr.0.iter())
                    .map(|(a, br)| a.cur() * br.cur())
                    .sum();
                let ABr_prev = ABr.1.prev();
                let ABr = ABr.1.cur();

                vec![q_ABr * (ABr - (ABr_prev + A_times_Br))]
            },
        );

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
pub struct FreivaldCircuit<F: Field, const NUM_INNER_COLS_A: usize, const NUM_INNER_COLS_B: usize> {
    // H x W
    A: Array<F>,
    // W x H
    B: Array<F>,
    // H x H
    C: Array<F>,
    H: usize,
    W: usize,
}

impl<F: Field, const NUM_INNER_COLS_A: usize, const NUM_INNER_COLS_B: usize>
    FreivaldCircuit<F, NUM_INNER_COLS_A, NUM_INNER_COLS_B>
{
    /// Create Freivald algorithm circuit
    #[allow(non_snake_case)]
    pub fn new(A: Array<F>, B: Array<F>, C: Array<F>, H: usize, W: usize) -> Self {
        Self { A, B, C, H, W }
    }
}

#[allow(non_snake_case)]
impl<F: Field, const NUM_INNER_COLS_A: usize, const NUM_INNER_COLS_B: usize> Circuit<F>
    for FreivaldCircuit<F, NUM_INNER_COLS_A, NUM_INNER_COLS_B>
{
    type Config = FreivaldConfig<NUM_INNER_COLS_A, NUM_INNER_COLS_B>;
    type FloorPlanner = V1;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        FreivaldConfig::<NUM_INNER_COLS_A, NUM_INNER_COLS_B>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let r = layouter.get_challenge(config.r);
        let powers_of_r = (0..NUM_INNER_COLS_B)
            .scan(Value::known(F::ONE), |r_power, _| {
                *r_power = *r_power * r;
                Some(*r_power)
            })
            .collect_vec()
            .into_iter()
            .rev()
            .collect_vec();
        let r_powered_to_num_inner_cols_b = powers_of_r[0] * r;

        let Br = layouter.assign_region(
            || "B • r",
            |mut region| {
                let mut Br_cells = vec![];

                for (row_idx, row) in self.B.transpose().vals.iter().enumerate() {
                    let chunked_row = row.iter().chunks(NUM_INNER_COLS_B);
                    let running_sum =
                        chunked_row
                            .into_iter()
                            .scan(Value::known(F::ZERO), |state, val| {
                                let curr_sum: Value<F> = val
                                    .into_iter()
                                    .zip(powers_of_r.iter())
                                    .map(|(v, r_power)| *v * *r_power)
                                    .reduce(|acc, v| acc + v)
                                    .unwrap();
                                *state = *state * r_powered_to_num_inner_cols_b + curr_sum;
                                Some(*state)
                            });
                    for (idx, (b_vals, br)) in row
                        .iter()
                        .chunks(NUM_INNER_COLS_B)
                        .into_iter()
                        .zip(running_sum)
                        .enumerate()
                    {
                        let offset = row_idx * self.B.num_cols() / NUM_INNER_COLS_B + idx;

                        b_vals
                            .into_iter()
                            .zip(config.B)
                            .map(|(b, b_col)| region.assign_advice(|| "", b_col, offset, || *b))
                            .collect::<Result<Vec<_>, Error>>()?;
                        let Br = region.assign_advice(|| "", config.Br, offset, || br)?;
                        if idx == self.B.num_cols() / NUM_INNER_COLS_B {
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
                    let running_sum = row
                        .iter()
                        .chunks(NUM_INNER_COLS_B)
                        .into_iter()
                        .scan(Value::known(F::ZERO), |state, val| {
                            let curr_sum: Value<F> = val
                                .into_iter()
                                .zip(powers_of_r.iter())
                                .map(|(v, r_power)| *v * *r_power)
                                .reduce(|acc, v| acc + v)
                                .unwrap();
                            *state = *state * r_powered_to_num_inner_cols_b + curr_sum;
                            Some(*state)
                        })
                        .collect_vec();
                    for (idx, (c_vals, cr)) in row
                        .iter()
                        .chunks(NUM_INNER_COLS_B)
                        .into_iter()
                        .zip(running_sum)
                        .enumerate()
                    {
                        let offset = row_idx * self.C.num_cols() / NUM_INNER_COLS_B + idx;

                        c_vals
                            .into_iter()
                            .zip(config.C)
                            .map(|(c, c_col)| region.assign_advice(|| "", c_col, offset, || *c))
                            .collect::<Result<Vec<_>, Error>>()?;
                        let Cr = region.assign_advice(|| "", config.Cr, offset, || cr)?;
                        if idx == self.C.num_cols() / NUM_INNER_COLS_B {
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
                    let running_sum = row
                        .into_iter()
                        .zip(Br_vals.iter())
                        .chunks(NUM_INNER_COLS_A)
                        .into_iter()
                        .scan(Value::known(F::ZERO), |state, vals| {
                            let curr_sum = vals
                                .into_iter()
                                .fold(Value::known(F::ZERO), |acc, (v, br)| acc + *v * br);
                            Some(*state + curr_sum)
                        })
                        .collect_vec();

                    for (idx, vals) in row
                        .iter()
                        .zip(Br.iter())
                        .chunks(NUM_INNER_COLS_A)
                        .into_iter()
                        .enumerate()
                    {
                        let offset = row_idx * self.A.num_cols() / NUM_INNER_COLS_A + idx;
                        let (a_vals, br_cells): (Vec<_>, Vec<_>) = vals.into_iter().unzip();
                        a_vals
                            .into_iter()
                            .zip(config.A)
                            .map(|(a, a_col)| region.assign_advice(|| "", a_col, offset, || a))
                            .collect::<Result<Vec<_>, Error>>()?;
                        br_cells
                            .into_iter()
                            .zip(config.ABr.0)
                            .map(|(br, br_col)| br.copy_advice(|| "", &mut region, br_col, offset))
                            .collect::<Result<Vec<_>, Error>>()?;
                        let ABr = region.assign_advice(
                            || "",
                            config.ABr.1,
                            offset,
                            || running_sum[idx],
                        )?;

                        if idx == 0 {
                            config.q_ABr.0.enable(&mut region, offset)?;
                        } else {
                            config.q_ABr.1.enable(&mut region, offset)?;
                        }

                        if idx == self.A.num_cols() / NUM_INNER_COLS_A - 1 {
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
