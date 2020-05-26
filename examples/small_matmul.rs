extern crate gpuarray as ga;

use ga::Context;
use ga::tensor::{Tensor, TensorMode};
use ga::array::Array;

fn main() {
    let ref ctx = Context::new();

    let (m,n,k): (usize,usize,usize) = (2,3,1);

    let a = Array::from_vec(vec![2,3], (0..m*n).map(|x| x as f32).collect());
    let b = Array::from_vec(vec![3,1], (0..n*k).map(|x| 1f32).collect());

    let a_gpu = Tensor::from_array(ctx, &a, TensorMode::In);
    let b_gpu = Tensor::from_array(ctx, &b, TensorMode::In);
    let c_gpu: Tensor<f32> = Tensor::new(ctx, vec![m, k], TensorMode::Mut);

    ga::matmul(ctx, &a_gpu, &b_gpu, &c_gpu);

    let c = c_gpu.get(ctx);

    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);
    println!("A*B = \n{:?}", c);
}
