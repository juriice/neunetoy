use std::iter::zip;

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
}

struct Neuron {
    bias: f64,
    weights: Vec<f64>,
    activation: Activation,
    error_delta: f64,
    bias_d: f64,
    weights_d: Vec<f64>,
    output: f64,
}

impl Neuron {
    fn new(inputs: usize, activation: Activation) -> Neuron {
        let neuron = Neuron {
            bias: rand::random_range(-0.1..0.1),
            weights: vec![rand::random_range(-0.33..0.33); inputs],
            activation: activation,
            error_delta: 0.0,
            bias_d: 0.0,
            weights_d: vec![0.0; inputs],
            output: 0.0,
        };
        neuron
    }

    fn calculate_output(&mut self, inputs: &Vec<f64>) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());

        let mut x: f64 = self.bias;
        for (w, i) in zip(&self.weights, inputs) {
            x += w * i;
        }

        match self.activation {
            Activation::Linear => self.output = x,
            Activation::Sigmoid => self.output = Self::sigmoid(x),
            Activation::Tanh => self.output = Self::tanh(x),
            Activation::Relu => self.output = Self::relu(x),
        };

        self.output
    }

    fn calculate_error_delta_output_layer(&mut self, target: f64) {
        self.error_delta = self.output - target;

        match self.activation {
            Activation::Linear => self.error_delta *= 1.0,
            Activation::Sigmoid => self.error_delta *= Self::sigmoid_derivative(self.output),
            Activation::Tanh => self.error_delta *= Self::tanh_derivative(self.output),
            Activation::Relu => self.error_delta *= Self::relu_derivative(self.output),
        }
    }

    fn calculate_error_delta_hidden_layer(&mut self, deltas: &Vec<f64>, weights: &Vec<f64>) {
        assert_eq!(deltas.len(), weights.len());

        self.error_delta = 0.0;
        for (d, w) in zip(deltas, weights) {
            self.error_delta += d * w;
        }

        match self.activation {
            Activation::Linear => self.error_delta *= 1.0,
            Activation::Sigmoid => self.error_delta *= Self::sigmoid_derivative(self.output),
            Activation::Tanh => self.error_delta *= Self::tanh_derivative(self.output),
            Activation::Relu => self.error_delta *= Self::relu_derivative(self.output),
        }
    }

    fn calculate_error_derivative(&mut self, inputs: &Vec<f64>) {
        assert_eq!(inputs.len(), self.weights_d.len());

        for (i, input) in inputs.iter().enumerate() {
            self.weights_d[i] += *input * self.error_delta;
        }

        self.bias_d += self.error_delta;
    }

    fn update_weights_and_bias(&mut self, learning_rate: f64, training_samples: usize) {
        for (i, weight) in self.weights.iter_mut().enumerate() {
            *weight -= learning_rate * self.weights_d[i] / training_samples as f64;
            self.weights_d[i] = 0.0;
        }

        self.bias -= learning_rate * self.bias_d / training_samples as f64;
        self.bias_d = 0.0;
    }

    fn get_weight_for_input(&self, input: usize) -> f64 { self.weights[input] }

    fn get_error_delta(&self) -> f64 { self.error_delta }

    fn get_output(&self) -> f64 { self.output }

    #[allow(dead_code)]
    fn set_bias(&mut self, bias: f64) { self.bias = bias; }

    #[allow(dead_code)]
    fn set_weight(&mut self, i: usize, weight: f64) { self.weights[i] = weight; }

    fn set_activation(&mut self, activation: Activation) { self.activation = activation; }

    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + x.exp()) }

    fn sigmoid_derivative(x: f64) -> f64 { let sigmoid = 1.0 / (1.0 + x.exp()); sigmoid * (1.0 - sigmoid) }

    pub fn tanh(x: f64) -> f64 { x.tanh() }

    pub fn tanh_derivative(x: f64) -> f64 { 1.0 - x.tanh().powi(2) }

    pub fn relu(x: f64) -> f64 { f64::max(0.0, x) }

    pub fn relu_derivative(x: f64) -> f64 { if x <= 0.0 { 0.0 } else { 1.0 } }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(neurons: usize, inputs: usize, activation: Activation) -> Self {
        let mut layer = Layer {neurons: Vec::new()};
        for _ in 0..neurons {
            layer.neurons.push(Neuron::new(inputs, activation));
        }
        layer
    }
}

pub struct Fnn {
    layers: Vec<Layer>,
    loss: f64,
}

impl Fnn {
    pub fn new(arch: &Vec<(usize, Activation)>, inputs: usize) -> Self {
        let mut fnn = Fnn {layers: Vec::new(), loss: 0.0};
        fnn.layers.push(Layer::new(arch[0].0, inputs, arch[0].1));
        for layer in 1..arch.len() {
            fnn.layers.push(Layer::new(arch[layer].0, arch[layer - 1].0, arch[layer].1));
        }
        fnn
    }

    #[allow(dead_code)]
    pub fn set_activation(&mut self, layer: usize, activation: Activation) {
        for neuron in 0..self.layers[layer].neurons.len() {
            self.layers[layer].neurons[neuron].set_activation(activation);
        }
    }

    pub fn feed_forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for layer in 0..self.layers.len() {
            let mut out : Vec<f64> = Vec::new();
            for neuron in 0..self.layers[layer].neurons.len() {
                match layer {
                    0 => out.push(self.layers[layer].neurons[neuron].calculate_output(input)),
                    _ => out.push(self.layers[layer].neurons[neuron].calculate_output(&output)),
                }
            }
            output = out;
        }
        output
    }

    pub fn calculate_error_delta(&mut self, targets: &Vec<f64>) {
        let output_layer = self.layers.len() - 1;
        assert_eq!(self.layers[output_layer].neurons.len(), targets.len());

        for neuron in 0..self.layers[output_layer].neurons.len() {
            self.layers[output_layer].neurons[neuron].calculate_error_delta_output_layer(targets[neuron])
        }

        for layer in (0..output_layer).rev() {
            for neuron in 0..self.layers[layer].neurons.len() {
                let input = neuron;
                let mut deltas: Vec<f64> = Vec::new();
                let mut weights: Vec<f64> = Vec::new();
                for neuron in 0..self.layers[layer + 1].neurons.len() {
                    deltas.push(self.layers[layer + 1].neurons[neuron].get_error_delta());
                    weights.push(self.layers[layer + 1].neurons[neuron].get_weight_for_input(input));
                }
                self.layers[layer].neurons[neuron].calculate_error_delta_hidden_layer(&deltas, &weights);
            }
        }
    }

    pub fn calculate_error_derivative(&mut self, inputs: &Vec<f64>) {
        for neuron in 0..self.layers[0].neurons.len() {
            self.layers[0].neurons[neuron].calculate_error_derivative(inputs);
        }

        for layer in 1..self.layers.len() {
            let mut outputs: Vec<f64> = Vec::new();
            for neuron in 0..self.layers[layer - 1].neurons.len() {
                outputs.push(self.layers[layer - 1].neurons[neuron].get_output());
            }
            for neuron in 0..self.layers[layer].neurons.len() {
                self.layers[layer].neurons[neuron].calculate_error_derivative(&outputs);
            }
        }
    }

    pub fn update_weights_and_bias(&mut self, learning_rate: f64, training_samples: usize) {
        for layer in 0..self.layers.len() {
            let mut outputs: Vec<f64> = Vec::new();
            for neuron in 0..self.layers[layer].neurons.len() {
                match layer {
                    0 => {
                        self.layers[layer].neurons[neuron].update_weights_and_bias(learning_rate, training_samples);
                    }
                    _ => {
                        outputs.push(self.layers[layer].neurons[neuron].get_output());
                    }
                }
            }
            if layer < self.layers.len() - 1 {
                for neuron in 0..self.layers[layer + 1].neurons.len() {
                    self.layers[layer + 1].neurons[neuron].update_weights_and_bias(learning_rate, training_samples);
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn set_weight(&mut self, layer: usize, neuron: usize, i: usize, weight: f64) {
        self.layers[layer].neurons[neuron].set_weight(i, weight);
    }

    #[allow(dead_code)]
    pub fn set_bias(&mut self, layer: usize, neuron: usize, bias: f64) {
        self.layers[layer].neurons[neuron].set_bias(bias);
    }

    #[allow(dead_code)]
    pub fn calculate_loss(&mut self, activation: Activation, output: f64, target: f64) {
        match activation {
            Activation::Linear => {
                self.loss += (output - target).powf(2.0);
            }
            _ => { panic!("{:?} not supported!", activation); }
        }
    }

    #[allow(dead_code)]
    pub fn get_loss(&mut self) -> f64 { let loss = self.loss; self.loss = 0.0; loss }
}
