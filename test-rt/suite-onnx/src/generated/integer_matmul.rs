use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct IntegerMatMul {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    signed: [bool; 2],
    present: [bool; 2],
    a: Vec<i32>,
    b: Vec<i32>,
    az: Vec<i32>,
    bz: Vec<i32>,
}
impl Arbitrary for IntegerMatMul {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        (1usize..4, 1usize..6, 1usize..12, 1usize..6, any::<[bool; 2]>(), any::<[bool; 2]>())
            .prop_flat_map(|(batch, m, k, n, signed, present)| {
                (
                    vec(0i32..256, batch * m * k),
                    vec(0i32..256, k * n),
                    vec(0i32..256, m),
                    vec(0i32..256, n),
                )
                    .prop_map(move |(mut a, mut b, mut az, mut bz)| {
                        if signed[0] {
                            a.iter_mut().chain(&mut az).for_each(|x| *x -= 128);
                        }
                        if signed[1] {
                            b.iter_mut().chain(&mut bz).for_each(|x| *x -= 128);
                        }
                        Self { batch, m, k, n, signed, present, a, b, az, bz }
                    })
            })
            .boxed()
    }
}
impl Test for IntegerMatMul {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let mut y = vec![0i32; self.batch * self.m * self.n];
        for batch in 0..self.batch {
            for m in 0..self.m {
                for n in 0..self.n {
                    for k in 0..self.k {
                        y[(batch * self.m + m) * self.n + n] += (self.a
                            [(batch * self.m + m) * self.k + k]
                            - if self.present[0] { self.az[m] } else { 0 })
                            * (self.b[k * self.n + n]
                                - if self.present[1] { self.bz[n] } else { 0 });
                    }
                }
            }
        }
        let args = vec![
            "a",
            "b",
            if self.present[0] { "az" } else { "" },
            if self.present[1] { "bz" } else { "" },
        ];
        check(
            NodeProto {
                op_type: "MatMulInteger".into(),
                input: args.into_iter().map(Into::into).collect(),
                output: vec!["y".into()],
                ..Default::default()
            },
            vec![
                ("a", byte_tensor(&[self.batch, self.m, self.k], &self.a, self.signed[0])),
                ("b", byte_tensor(&[self.k, self.n], &self.b, self.signed[1])),
            ],
            vec![
                ("az", byte_tensor(&[self.m], &self.az, self.signed[0])),
                ("bz", byte_tensor(&[self.n], &self.bz, self.signed[1])),
            ],
            vec![Tensor::from_shape(&[self.batch, self.m, self.n], &y)?],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<IntegerMatMul>("broadcast_signedness_zero_points", ());
    s
}
