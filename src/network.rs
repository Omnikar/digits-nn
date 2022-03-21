use std::sync::{Arc, RwLock};

pub type Matrix = nalgebra::base::DMatrix<f64>;
pub type Vector = nalgebra::base::DVector<f64>;

pub struct Network {
    pub conf: Arc<RwLock<NetConf>>,
    pub state: NetState,
}

#[derive(Clone)]
pub struct NetConf {
    layers: Vec<Layer>,
    cost: Cost,
}

#[derive(Clone)]
pub struct NetState {
    layers: Vec<Vector>,
}

#[derive(Clone)]
struct Layer {
    weights: Matrix,
    biases: Vector,
    activation: Activation,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let conf = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut layers = layers
                .windows(2)
                .map(|layers| {
                    let (input, output) = (layers[0], layers[1]);
                    let weights = Matrix::from_fn(output, input, |_, _| {
                        let mag = (2.0 / input as f64).sqrt();
                        rng.gen_range(-mag..mag)
                    });
                    let biases = Vector::from_element(output, 0.0);
                    Layer {
                        weights,
                        biases,
                        activation: Activation::RELU,
                    }
                })
                .collect::<Vec<_>>();
            layers.last_mut().unwrap().activation = Activation::SOFTMAX;
            NetConf {
                layers,
                cost: Cost::CAT_CE,
            }
        };
        let state = {
            let layers = layers
                .iter()
                .copied()
                .map(|n| Vector::from_element(n, 0.0))
                .collect();
            NetState { layers }
        };
        let conf = Arc::new(RwLock::new(conf));
        Self { conf, state }
    }

    pub fn from_conf(conf: NetConf) -> Self {
        let state = {
            let layers = std::iter::once(conf.layers[0].weights.ncols())
                .chain(conf.layers.iter().map(|layer| layer.biases.nrows()))
                .map(|n| Vector::from_element(n, 0.0))
                .collect();
            NetState { layers }
        };
        let conf = Arc::new(RwLock::new(conf));
        Self { conf, state }
    }

    pub fn process(&mut self, input: &Vector) {
        self.state.layers[0].copy_from(input);
        for (layer, (inp_i, out_i)) in self
            .conf
            .read()
            .unwrap()
            .layers
            .iter()
            .zip((0..self.state.layers.len()).zip(1..self.state.layers.len()))
        {
            self.state.layers[out_i] = layer.calculate(&self.state.layers[inp_i]);
        }
    }

    pub fn output(&self) -> &Vector {
        self.state.layers.last().unwrap()
    }

    pub fn cost(&self, expected: &Vector) -> f64 {
        (self.conf.read().unwrap().cost.fun)(self.output(), expected)
    }

    pub fn gradient(&self, expected: &Vector) -> Vector {
        let conf = self.conf.read().unwrap();
        let mut node_derivs = Vec::with_capacity(conf.layers.len());
        node_derivs.push((conf.cost.deriv)(self.output(), expected));
        let act_derivs = conf
            .layers
            .iter()
            .zip(self.state.layers.iter())
            .map(|(layer, prev)| layer.deriv(prev))
            .collect::<Vec<_>>();
        for (i, (layer, act_deriv)) in conf.layers.iter().zip(act_derivs.iter()).rev().enumerate() {
            // let next = &layer.weights * act_deriv * &node_derivs[i];
            let next = layer.weights.transpose() * act_deriv * &node_derivs[i];
            // let mut next = act_deriv * &node_derivs[i];
            // next = &layer.weights * next;
            node_derivs.push(next);
        }

        self.state
            .layers
            .iter()
            .zip(act_derivs.iter())
            .rev()
            .zip(node_derivs.iter())
            .flat_map(|((prev, act_deriv), layer_deriv)| {
                let bias_deriv = act_deriv * layer_deriv;
                let weight_deriv = &bias_deriv * prev.transpose();
                weight_deriv
                    .iter()
                    .copied()
                    .chain(bias_deriv.iter().copied())
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect::<Vec<_>>()
            .into()
    }
}

impl NetConf {
    pub fn flatten(&self) -> Vector {
        let data = self
            .layers
            .iter()
            .rev()
            .flat_map(|layer| {
                layer
                    .weights
                    .iter()
                    .copied()
                    .chain(layer.biases.iter().copied())
            })
            .collect::<Vec<_>>();
        Vector::from_column_slice(&data)
    }

    pub fn load_iter(&mut self, iter: impl Iterator<Item = f64>) {
        self.layers
            .iter_mut()
            .rev()
            .flat_map(|layer| layer.weights.iter_mut().chain(layer.biases.iter_mut()))
            .zip(iter)
            .for_each(|(n, val)| *n = val);
    }
}

impl Layer {
    fn calculate(&self, prev: &Vector) -> Vector {
        (self.activation.fun)(&self.weights * prev + &self.biases)
    }

    fn deriv(&self, prev: &Vector) -> Matrix {
        (self.activation.deriv)(&(&self.weights * prev + &self.biases))
    }
}

use funcs::*;
mod funcs {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct Activation {
        pub fun: fn(Vector) -> Vector,
        pub deriv: fn(&Vector) -> Matrix,
    }

    impl Activation {
        pub const RELU: Self = Self {
            fun: Self::relu,
            deriv: Self::relu_deriv,
        };
        pub const SOFTMAX: Self = Self {
            fun: Self::softmax,
            deriv: Self::softmax_deriv,
        };

        fn relu(mut input: Vector) -> Vector {
            input.apply(|n| *n = n.max(0.0));
            input
        }
        fn relu_deriv(input: &Vector) -> Matrix {
            let mut output = Matrix::from_element(input.nrows(), input.nrows(), 0.0);
            input
                .iter()
                .enumerate()
                .for_each(|(i, n)| output[(i, i)] = (*n >= 0.0) as u64 as f64);
            output
        }

        fn softmax(mut input: Vector) -> Vector {
            input.apply(|n| *n = n.exp());
            let sum = input.sum();
            input.apply(|n| *n /= sum);
            input
        }
        fn softmax_deriv(input: &Vector) -> Matrix {
            let mut output = Matrix::from_element(input.nrows(), input.nrows(), 0.0);
            input
                .iter()
                .enumerate()
                .for_each(|(i, n)| output[(i, i)] = n.exp());
            let sum = output.sum();
            output.map_with_location(|r, c, n| {
                (n * sum - output[(r, r)] * output[(c, c)]) / sum.powi(2)
            })
        }
    }

    #[derive(Clone, Copy)]
    pub struct Cost {
        pub fun: fn(&Vector, &Vector) -> f64,
        pub deriv: fn(&Vector, &Vector) -> Vector,
    }

    impl Cost {
        pub const SQUARE: Self = Self {
            fun: Self::square,
            deriv: Self::square_deriv,
        };
        pub const CAT_CE: Self = Self {
            fun: Self::cat_ce,
            deriv: Self::cat_ce_deriv,
        };

        fn square(actual: &Vector, expected: &Vector) -> f64 {
            let mut error = actual - expected;
            error.iter_mut().for_each(|n| *n *= *n);
            error.sum()
        }
        fn square_deriv(actual: &Vector, expected: &Vector) -> Vector {
            let mut error = actual - expected;
            error.iter_mut().for_each(|n| *n *= 2.0);
            error
        }

        fn cat_ce(actual: &Vector, expected: &Vector) -> f64 {
            let mut cost = actual.clone().apply_into(|n| *n = n.ln());
            cost.component_mul_assign(expected);
            -cost.sum()
        }
        fn cat_ce_deriv(actual: &Vector, expected: &Vector) -> Vector {
            -expected.component_div(actual)
        }
    }
}
