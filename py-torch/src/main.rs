//! # Example of how to use a PyTorch exported script module in Rust
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-15
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** None known at this time.
//! - **Todo:** Further development tasks to be determined.


use tch::{jit, Tensor, Device, Kind};
use std::error::Error;
use std::fs::File;
use std::io::{Write, BufWriter};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: example-app <path-to-exported-script-module>");
        std::process::exit(1);
    }

    let module_path = &args[1];
    let module = jit::CModule::load(module_path)?;

    let num_steps = 40000;
    let mut data = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        data.push(i as f32 * 0.001);
    }

    let input_tensor = Tensor::from_slice(&data).view([num_steps as i64, 1]);
    // input_tensor.print();
    println!("Input: {:?}", &input_tensor);

    let input_ivalue = tch::IValue::Tensor(input_tensor);
    let output_ivalue = module.forward_is(&[input_ivalue])?;
    let output = if let tch::IValue::Tensor(output) = output_ivalue {
        output
    } else {
        return Err("Expected Tensor".into());
    };

    let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
    println!("Output: {:?}", &output);

    // Extracting data from the output tensor
    // Extracting columns from the output tensor
    let output_data_1: Vec<f32> = (0..output.size()[0])
        .map(|i| output.double_value(&[i as i64, 0]) as f32) // First column
        .collect();
    let output_data_2: Vec<f32> = (0..output.size()[0])
        .map(|i| output.double_value(&[i as i64, 1]) as f32) // Second column
        .collect();

    // Plotting
    let root = BitMapBackend::new("output_plot.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Output Data Plot", ("sans-serif", 40))
        .margin(5)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0f32..40f32, // x-axis now goes from 0 to 40
            output_data_1.iter().cloned().chain(output_data_2.iter().cloned()).fold(f32::INFINITY, f32::min)..output_data_1.iter().cloned().chain(output_data_2.iter().cloned()).fold(f32::NEG_INFINITY, f32::max)
        )?;

    chart.configure_mesh().draw()?;

    // Plotting the first series
    let plot_data_1: Vec<(f32, f32)> = data.iter().cloned().zip(output_data_1.into_iter()).collect();
    chart.draw_series(LineSeries::new(plot_data_1, &RED))?;

    // Plotting the second series
    let plot_data_2: Vec<(f32, f32)> = data.iter().cloned().zip(output_data_2.into_iter()).collect();
    chart.draw_series(LineSeries::new(plot_data_2, &BLUE))?;

    root.present()?;
    println!("Plot saved as output_plot.png");

    // Writing the output to a CSV file
    write_tensor_to_csv(&output, "output.csv")?;

    Ok(())
}


/// Writes a tensor to a CSV file
fn write_tensor_to_csv(output: &Tensor, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure the tensor is on CPU and in Float format for easy handling
    let output = output.to_device(Device::Cpu).to_kind(tch::Kind::Float);

    // Create a buffered writer for the output file
    let mut file = BufWriter::new(File::create(file_name)?);

    // Get the dimensions of the tensor
    let sizes = output.size();
    let num_rows = sizes[0] as usize;
    let num_cols = sizes[1] as usize;

    // Write each element to the CSV file
    for i in 0..num_rows {
        for j in 0..num_cols {
            write!(file, "{:.6}", output.double_value(&[i as i64, j as i64]))?;
            if j != num_cols - 1 {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }

    println!("Output saved as {}", file_name);
    Ok(())
}
