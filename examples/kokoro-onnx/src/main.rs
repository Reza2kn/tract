use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayViewD;
use ndarray_npy::NpzReader;
use ndarray_npy::NpzWriter;
use std::fs::File;
use std::time::Instant;
use tract::prelude::*;

fn main() -> Result<()> {
    let stabilize_phase = std::env::args().any(|a| a == "--stabilize-phase");
    let args: Vec<_> = std::env::args().filter(|a| a != "--stabilize-phase").collect();
    ensure!(
        args.len() >= 4 && args.len().is_multiple_of(2),
        "Usage: {} [--stabilize-phase] MODEL.onnx INPUTS.npz OUTPUT.npz [INPUTS.npz OUTPUT.npz ...]",
        args[0]
    );
    let options = OnnxOptions::new()
        .ignore_value_info(true)
        .assertion("sequence_length >= 2")
        .assertion("sequence_length <= 512");
    let mut model = tract::onnx()?.load_with_options(&args[1], options)?;
    model.set_output_fact(0, InferenceFact::empty()?)?;
    let mut model = model.into_model()?;
    if stabilize_phase {
        model.transform("kokoro_stabilize_phase")?;
        println!("Phase stabilization enabled: zero complex bins have phase zero");
    }
    let model = model.into_runnable()?;
    for paths in args[2..].chunks_exact(2) {
        let mut inputs = NpzReader::new(File::open(&paths[0])?)?;
        let tokens: Array2<i64> = inputs.by_name("tokens")?;
        let style: Array2<f32> = inputs.by_name("style")?;
        let speed: Array1<f32> = inputs.by_name("speed")?;
        ensure!(
            tokens.shape()[0] == 1 && (2..=512).contains(&tokens.shape()[1]),
            "Expected 1 x (2..512) tokens, including boundary tokens"
        );
        ensure!(
            style.shape() == [1, 256] && style.iter().all(|v| v.is_finite()),
            "Expected a finite 1 x 256 voice style"
        );
        ensure!(
            speed.len() == 1 && speed[0].is_finite() && speed[0] > 0.,
            "Speed must be finite and positive"
        );
        let tokens = Tensor::from_slice(
            tokens.shape(),
            tokens.as_slice().context("Non-contiguous tokens")?,
        )?;
        let style =
            Tensor::from_slice(style.shape(), style.as_slice().context("Non-contiguous style")?)?;
        let speed =
            Tensor::from_slice(speed.shape(), speed.as_slice().context("Non-contiguous speed")?)?;
        let start = Instant::now();
        let output = model.run([tokens, style, speed])?;
        let elapsed = start.elapsed();
        let (shape, audio) = output[0].as_shape_and_slice::<f32>()?;
        ensure!(audio.iter().all(|v| v.is_finite()), "Model produced non-finite audio");
        let mut bundle = NpzWriter::new_compressed(File::create(&paths[1])?);
        bundle.add_array("audio", &ArrayViewD::from_shape(shape, audio)?)?;
        bundle.finish()?;
        println!(
            "{} samples at 24000 Hz ({:.2} seconds), inference {:.3} seconds",
            audio.len(),
            audio.len() as f64 / 24000.,
            elapsed.as_secs_f64()
        );
    }
    Ok(())
}
