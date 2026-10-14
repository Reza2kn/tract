use super::*;
use infra::Test;
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct FusedMatMul {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    flags: [bool; 4],
    a: Vec<f32>,
    b: Vec<f32>,
}
impl Arbitrary for FusedMatMul {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_: ()) -> Self::Strategy {
        (1usize..4, 1usize..5, 1usize..6, 1usize..5, any::<[bool; 4]>())
            .prop_flat_map(|(batch, m, k, n, flags)| {
                (vec(-8i32..9, batch * m * k), vec(-8i32..9, batch * k * n)).prop_map(
                    move |(a, b)| Self {
                        batch,
                        m,
                        k,
                        n,
                        flags,
                        a: a.into_iter().map(|x| x as f32 / 4.).collect(),
                        b: b.into_iter().map(|x| x as f32 / 4.).collect(),
                    },
                )
            })
            .boxed()
    }
}
fn permuted(
    batch: usize,
    rows: usize,
    cols: usize,
    values: &[f32],
    transpose: bool,
    batch_transpose: bool,
) -> Tensor {
    let array = Tensor::from_shape(&[batch, rows, cols], values)
        .unwrap()
        .to_plain_array_view::<f32>()
        .unwrap()
        .to_owned();
    let array = if transpose { array.permuted_axes(vec![0, 2, 1]) } else { array };
    if batch_transpose {
        array.permuted_axes(vec![1, 0, 2]).into_tensor()
    } else {
        array.into_tensor()
    }
}
impl Test for FusedMatMul {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let mut y = vec![0.; self.batch * self.m * self.n];
        for batch in 0..self.batch {
            for m in 0..self.m {
                for n in 0..self.n {
                    for k in 0..self.k {
                        y[(batch * self.m + m) * self.n + n] += self.a
                            [(batch * self.m + m) * self.k + k]
                            * self.b[(batch * self.k + k) * self.n + n]
                            * 0.5;
                    }
                }
            }
        }
        check(
            NodeProto {
                op_type: "FusedMatMul".into(),
                domain: "com.microsoft".into(),
                input: vec!["a".into(), "b".into()],
                output: vec!["y".into()],
                attribute: vec![
                    int_attr("transA", self.flags[0] as i64),
                    int_attr("transB", self.flags[1] as i64),
                    int_attr("transBatchA", self.flags[2] as i64),
                    int_attr("transBatchB", self.flags[3] as i64),
                    float_attr("alpha", 0.5),
                ],
                ..Default::default()
            },
            vec![
                ("a", permuted(self.batch, self.m, self.k, &self.a, self.flags[0], self.flags[2])),
                ("b", permuted(self.batch, self.k, self.n, &self.b, self.flags[1], self.flags[3])),
            ],
            vec![],
            vec![Tensor::from_shape(&[self.batch, self.m, self.n], &y)?],
            runtime,
            Approximation::Exact,
        )
    }
}
pub fn suite() -> TestSuite {
    let mut s = TestSuite::default();
    s.add_arbitrary::<FusedMatMul>("transpose_and_batch", ());
    for (ix, name) in ["dot", "vector_matrix", "matrix_vector"].iter().enumerate() {
        s.add(name, VectorMatMul(ix));
    }
    s
}

#[derive(Clone, Debug)]
struct VectorMatMul(usize);
impl Test for VectorMatMul {
    fn run_with_approx(&self, _: &str, runtime: &dyn Runtime, _: Approximation) -> TractResult<()> {
        let a = if self.0 == 2 {
            tensor3(&[[[1f32, 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
        } else {
            tensor1(&[1f32, 2., 3.])
        };
        let b = if self.0 == 1 {
            tensor3(&[[[1f32, 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]])
        } else {
            tensor1(&[1f32, 2., 3.])
        };
        let expected = match self.0 {
            0 => tensor0(14f32),
            1 => tensor2(&[[22f32, 28.], [58., 64.]]),
            _ => tensor2(&[[14f32, 32.], [50., 68.]]),
        };
        check(
            NodeProto {
                op_type: "FusedMatMul".into(),
                domain: "com.microsoft".into(),
                input: vec!["a".into(), "b".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
            vec![("a", a), ("b", b)],
            vec![],
            vec![expected],
            runtime,
            Approximation::Exact,
        )
    }
}
