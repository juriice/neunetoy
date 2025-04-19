mod fnn;
use fnn::Fnn;
use fnn::Activation;

use std::cmp;

#[allow(dead_code)]
/// training_range: (range start, range end, number of training samples)
/// test_range:     (range start, range end, number of test samples)
fn example_linear_regression_linear_data(training_range: (f64, f64, usize), test_range: (f64, f64, usize)) {
    const EPOCHS: usize = 100_000;
    const EPOCHS_SUPPRESS_LOG: usize = EPOCHS / 10;
    const LEARNING_RATE: f64 = 0.000_001;

     // Create random input data from the given range
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    for _ in 1..=training_range.2 {
        let input: Vec<f64> = vec![rand::random_range(training_range.0..training_range.1)];
        inputs.push(input);
    }

    println!("\nTrain to do: y = 5x + 100");
    let y = |x: f64| -> f64 {5.0*x + 100.0};
    let mut targets: Vec<Vec<f64>> = Vec::new();
    for input in inputs.iter() {
        let target: Vec<f64> = vec![y(input[0])];
        targets.push(target);
    }

    let arch: Vec<(usize, Activation)> = vec![(1, Activation::Linear)];
    println!("Create an FNN with: {:?}", arch);

    let mut fnn = Fnn::new(&arch, inputs[0].len());

    println!("Start the training for {} samples in {} epochs:\n", inputs.len(), EPOCHS);
    for epoch in 1..=EPOCHS {
        for (i, input) in inputs.iter().enumerate() {
            let output = fnn.feed_forward(input);
            fnn.calculate_loss(Activation::Linear, output[0], targets[i][0]);

            // Backpropagation
            fnn.calculate_error_delta(&targets[i]);
            fnn.calculate_error_derivative(input);
            // Stochastic Gradient Descent
            fnn.update_weights_and_bias(LEARNING_RATE, 1);
        }
        // Batch Gradient Descent
        // fnn.update_weights_and_bias(LEARNING_RATE, inputs.len());

        // SGD is giving much better result for this case
        let loss = fnn.get_loss() / inputs.len() as f64;
        if epoch % EPOCHS_SUPPRESS_LOG == 0 { println!("Epoch: {} Loss: {:.8}", epoch, loss) }
    }

    println!("\nTest the FNN:");
    for _ in 1..test_range.2 {
        let input: Vec<f64> = vec![rand::random_range(test_range.0..test_range.1)];
        let target: u64 = y(input[0]) as u64;
        let output: u64 = fnn.feed_forward(&input)[0] as u64;
        let error: u64 = cmp::max(target, output) - cmp::min(target, output);
        println!("Target: {} Output: {} Error: {} {:.8}%", target, output, error, (error as f64 / target as f64) * 100.0);
    }
}

#[allow(dead_code)]
/// training_range: (range start, range end, number of training samples)
/// test_range:     (range start, range end, number of test samples)
fn example_linear_regression_nonlinear_data(training_range: (f64, f64, usize), test_range: (f64, f64, usize)) {
    const EPOCHS: usize = 200_000;
    const EPOCHS_SUPPRESS_LOG: usize = EPOCHS / 10;
    const LEARNING_RATE: f64 = 0.000_000_1;

    // Create random input data from the given range
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    for _ in 1..=training_range.2 {
        let input: Vec<f64> = vec![rand::random_range(training_range.0..training_range.1)];
        inputs.push(input);
    }

    println!("\nTrain to do: y = 2x + rand(1..100)");
    let y = |x: f64| -> f64 {2.0*x + rand::random_range(1.0..100.0)};
    let mut targets: Vec<Vec<f64>> = Vec::new();
    for input in inputs.iter() {
        let target: Vec<f64> = vec![y(input[0])];
        targets.push(target);
    }

    let arch: Vec<(usize, Activation)> = vec![(32, Activation::Relu), (16, Activation::Relu), (1, Activation::Linear)];
    println!("Create an FNN with: {:?}", arch);

    let mut fnn = Fnn::new(&arch, inputs[0].len());

    println!("Start the training for {} samples in {} epochs:\n", inputs.len(), EPOCHS);
    for epoch in 1..=EPOCHS {
        for (i, input) in inputs.iter().enumerate() {
            let output = fnn.feed_forward(input);
            fnn.calculate_loss(Activation::Linear, output[0], targets[i][0]);

            // Backpropagation
            fnn.calculate_error_delta(&targets[i]);
            fnn.calculate_error_derivative(input);
            // Stochastic Gradient Descent
            fnn.update_weights_and_bias(LEARNING_RATE, 1);
        }
        // Batch Gradient Descent
        // fnn.update_weights_and_bias(LEARNING_RATE, inputs.len());

        // SGD is giving much better result for this case
        let loss = fnn.get_loss() / inputs.len() as f64;
        if epoch % EPOCHS_SUPPRESS_LOG == 0 { println!("Epoch: {} Loss: {:.2}", epoch, loss) }
    }

    println!("\nTest the FNN:");
    for _ in 1..test_range.2 {
        let input: Vec<f64> = vec![rand::random_range(test_range.0..test_range.1)];
        let target: u64 = y(input[0]) as u64;
        let output: u64 = fnn.feed_forward(&input)[0] as u64;
        let error: u64 = cmp::max(target, output) - cmp::min(target, output);
        println!("Target: {} Output: {} Error: {} {:.2}%", target, output, error, (error as f64 / target as f64) * 100.0);
    }
}

#[allow(dead_code)]
fn example_linear_regression_xor_data() {
    const EPOCHS: usize = 100_000;
    const EPOCHS_SUPPRESS_LOG: usize = EPOCHS / 10;
    const LEARNING_RATE: f64 = 0.001;

    // XOR inputs and targets
    let inputs: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let targets: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    println!("\nTrain to do: XOR");

    let arch: Vec<(usize, Activation)> = vec![(2, Activation::Relu), (1, Activation::Linear)];
    println!("Create an FNN with: {:?}", arch);

    let mut fnn = Fnn::new(&arch, inputs[0].len());

    // The default random initialization seams to be too low and the network can't be train,
    // or it needs many attempts to get a good results
    println!("Patch the Neurons to learn faster ;)\n");
    fnn.set_bias(0, 0, -6.0);
    fnn.set_bias(0, 1, 1.0);
    fnn.set_bias(1, 0, 3.0);
    fnn.set_weight(0, 0, 0, 10.0);
    fnn.set_weight(0, 0, 1, 10.0);
    fnn.set_weight(0, 1, 0, 3.0);
    fnn.set_weight(0, 1, 1, 3.0);
    fnn.set_weight(1, 0, 0, -5.0);
    fnn.set_weight(1, 0, 1, -5.0);

    println!("Start the training for {} samples in {} epochs:\n", inputs.len(), EPOCHS);
    for epoch in 1..=EPOCHS {
        for (i, input) in inputs.iter().enumerate() {
            let output = fnn.feed_forward(input);
            fnn.calculate_loss(Activation::Linear, output[0], targets[i][0]);

            // Backpropagation
            fnn.calculate_error_delta(&targets[i]);
            fnn.calculate_error_derivative(input);
            // Stochastic Gradient Descent
            // fnn.update_weights_and_bias(LEARNING_RATE, 1);
        }
        // Batch Gradient Descent
        fnn.update_weights_and_bias(LEARNING_RATE, inputs.len());

        // Both SGD and BGD are giving same good result for this case
        let loss = fnn.get_loss() / inputs.len() as f64;
        if epoch % EPOCHS_SUPPRESS_LOG == 0 { println!("Epoch: {} Loss: {:.2}", epoch, loss) }
    }

    println!("\nTest the FNN:");
    for (i, input) in inputs.iter().enumerate() {
        println!("Target: {} Output: {:.2}", targets[i][0], fnn.feed_forward(&input)[0]);
    }
}

fn main() {
    example_linear_regression_linear_data((0.0, 1_000.0, 1_000), (1_000_000_000.0, 2_000_000_000.0, 10));

    example_linear_regression_nonlinear_data((0.0, 100.0, 100), (100.0, 200.0, 10));

    example_linear_regression_xor_data();
}
