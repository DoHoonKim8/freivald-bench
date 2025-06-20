use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use halo2_gadgets::mat_mul::{
    dot_product::DotProductCircuit,
    freivalds::{Array, FreivaldCircuit},
};
use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::{
    arithmetic::log2_floor,
    circuit::Value,
    halo2curves::bn256::{Bn256, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, ProvingKey},
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use rand::rngs::OsRng;

struct ParamsKZGCache {
    k: u32,
    params: Option<ParamsKZG<Bn256>>,
}

impl ParamsKZGCache {
    fn fetch_params<'a>(&'a mut self, k: u32) -> &'a ParamsKZG<Bn256> {
        if let Some(_) = self.params {
            if self.k != k {
                let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
                self.k = k;
                self.params = Some(params);
            }
        } else {
            let params: ParamsKZG<Bn256> = ParamsKZG::new(k);
            self.k = k;
            self.params = Some(params);
        }
        &self.params.as_ref().unwrap()
    }
}

/// [HxW][Wx1]
#[allow(non_snake_case)]
fn freivalds_setup<'a>(
    H: usize,
    W: usize,
    cache: &'a mut ParamsKZGCache,
) -> (
    &'a ParamsKZG<Bn256>,
    FreivaldCircuit<Fr>,
    ProvingKey<G1Affine>,
) {
    let K = log2_floor(W * H - 1) + 1;
    let params = cache.fetch_params(K);

    let A = Array::<Fr>::rand(&mut OsRng, W, H);
    let B = Array::<Fr>::rand(&mut OsRng, 1, W);
    let C = A.mat_mul(&B);

    // Sanity check in plaintext
    {
        let r: Vec<Value<Fr>> = vec![Value::known(Fr::random(OsRng))];

        let b = B.vec_mul(&r);
        let c = C.vec_mul(&r);
        let Ab = A.vec_mul(&b);
        assert!(Ab.len() == H && c.len() == H);

        Ab.iter()
            .zip(c.iter())
            .for_each(|(Ab, c)| (Ab - c).assert_if_known(|v| *v == Fr::ZERO));
    }

    let circuit = FreivaldCircuit::new(A, B, C, H, W);
    let vk = keygen_vk(params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(params, vk, &circuit).expect("keygen_pk should not fail");
    (params, circuit, pk)
}

#[allow(non_snake_case)]
fn run_freivalds(
    params: &ParamsKZG<Bn256>,
    circuit: &FreivaldCircuit<Fr>,
    pk: &ProvingKey<G1Affine>,
) {
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        params,
        pk,
        &[circuit.clone()],
        &[&[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");

    transcript.finalize();
}

/// `num_dot_products` * [1xN][Nx1]
#[allow(non_snake_case)]
fn dot_product_setup(
    num_dot_products: u32,
    N: usize,
    cache: &mut ParamsKZGCache,
) -> (
    &ParamsKZG<Bn256>,
    DotProductCircuit<Fr>,
    ProvingKey<G1Affine>,
) {
    // current dot product circuit divides vector sets into 3 chunks
    let K = log2_floor(
        ((N as u32 * num_dot_products.div_ceil(3)) - 1)
            .try_into()
            .unwrap(),
    ) + 1;
    let params = cache.fetch_params(K);
    let values = (0..num_dot_products)
        .map(|_| {
            let a = (0..N)
                .map(|_| Value::known(Fr::random(&mut OsRng)))
                .collect();
            let b = (0..N)
                .map(|_| Value::known(Fr::random(&mut OsRng)))
                .collect();
            (a, b)
        })
        .collect();
    let circuit = DotProductCircuit::new(values);
    let vk = keygen_vk(params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(params, vk, &circuit).expect("keygen_pk should not fail");
    (params, circuit, pk)
}

#[allow(non_snake_case)]
fn run_dot_product(
    params: &ParamsKZG<Bn256>,
    circuit: &DotProductCircuit<Fr>,
    pk: &ProvingKey<G1Affine>,
) {
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        params,
        pk,
        &[circuit.clone()],
        &[&[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");

    transcript.finalize();
}

#[allow(non_snake_case)]
fn bench_squared_mat_muls(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared mat mul");
    group.sample_size(10);
    let max_dimension = 10000;
    let step = 1000;

    let mut freivalds_params_cache = ParamsKZGCache { k: 0, params: None };
    let mut dot_product_params_cache = ParamsKZGCache { k: 0, params: None };
    let num_dot_products = 3;

    (0..=max_dimension)
        .into_iter()
        .step_by(step)
        .skip(1)
        .for_each(|N: usize| {
            let (params, circuit, pk) =
                freivalds_setup(num_dot_products, N, &mut freivalds_params_cache);
            group.bench_with_input(
                BenchmarkId::new("run_freivalds", N),
                &(params, circuit, pk),
                |b, (params, circuit, pk)| b.iter(|| run_freivalds(params, circuit, pk)),
            );

            let (params, circuit, pk) =
                dot_product_setup(num_dot_products as u32, N, &mut dot_product_params_cache);
            group.bench_with_input(
                BenchmarkId::new("run_dot_product", N),
                &(params, circuit, pk),
                |b, (params, circuit, pk)| b.iter(|| run_dot_product(params, circuit, pk)),
            );
        });
    group.finish();
}

criterion_group!(benches, bench_squared_mat_muls);
criterion_main!(benches);
