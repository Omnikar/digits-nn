mod config;
mod loader;
mod network;
mod thread;

use rand::seq::SliceRandom;

const IMAGE_SIZE: usize = 28;

fn main() -> anyhow::Result<()> {
    let config = config::load_config()?;

    let labels = loader::load_labels(&config.data.train.labels)?;
    let images = loader::load_images(&config.data.train.images)?;

    let mut layers = Vec::with_capacity(config.h_layers.len() + 2);
    layers.push(IMAGE_SIZE.pow(2));
    layers.extend_from_slice(&config.h_layers);
    layers.push(10);

    let mut nn = network::Network::new(&layers);
    if let Ok(s) = std::fs::read("network") {
        nn.conf.write().unwrap().load_iter(s.chunks(8).map(|chunk| {
            let mut bytes = [0; 8];
            bytes.copy_from_slice(chunk);
            f64::from_be_bytes(bytes)
        }));
    }

    let pool = thread::ThreadPool::new(16, || network::Network {
        conf: std::sync::Arc::clone(&nn.conf),
        state: nn.state.clone(),
    });

    let nbatches = (labels.len() + config.batch_size - 1) / config.batch_size;
    let mut momentum = 0.0 * nn.conf.read().unwrap().flatten();

    if std::env::args().nth(1).as_deref() != Some("--test") {
        for _ in 0..config.epochs {
            let mut ordering = (0..labels.len()).collect::<Vec<_>>();
            ordering.shuffle(&mut rand::thread_rng());

            print!("\n\n");

            let mut overall_avg_cost = 0.0;
            for (i, batch) in ordering.chunks(config.batch_size).enumerate() {
                batch.iter().copied().for_each(|i| {
                    let label = labels[i];
                    let image = images[i];
                    pool.execute(move |nn: &mut network::Network| {
                        let input_vector = image.into();
                        let expected = expected(label);
                        nn.process(&input_vector);
                        Some((nn.gradient(&expected), nn.cost(&expected)))
                    });
                });
                let (gradients, costs): (Vec<_>, Vec<_>) = pool.results(batch.len()).unzip();
                let len = gradients.len() as f64;
                let avg = gradients
                    .into_iter()
                    .sum::<network::Vector>()
                    .map(|n| n / len);
                let avg_cost = costs.into_iter().sum::<f64>() / len;
                let scaled_gradient = config.learning_rate * avg;
                momentum = config.momentum_decay * momentum + &scaled_gradient;
                let step = config.momentum_decay * &momentum + scaled_gradient;
                let mut conf = nn.conf.write().unwrap();
                let flattened = conf.flatten();
                conf.load_iter((flattened - &step).iter().copied());
                overall_avg_cost += avg_cost / nbatches as f64;
                println!(
                    "\x1b[2Abatch {}/{} complete; avg cost: {}\n{}\x1b[K",
                    i + 1,
                    nbatches,
                    avg_cost,
                    gen_bar((avg_cost * 100.0) as usize),
                );
            }

            println!("overall avg cost: {}", overall_avg_cost);

            save(&nn.conf.read().unwrap())?;
        }
    } else {
        let test_labels = loader::load_labels(&config.data.test.labels)?;
        let test_images = loader::load_images(&config.data.test.images)?;

        for (label, image) in test_labels.iter().zip(test_images.iter()).take(10) {
            let input_vector = (*image).into();
            nn.process(&input_vector);
            print_info(*label, image, &nn);
        }

        let mut avg_cost = 0.0;
        let mut accuracy = 0.0;
        let count = test_labels.len() as f64;
        for (label, image) in test_labels.into_iter().zip(test_images) {
            let input_vector = image.into();
            let expected = expected(label);
            nn.process(&input_vector);
            avg_cost += nn.cost(&expected) / count;
            let output = nn
                .output()
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            accuracy += 100.0 * (output == label as usize) as u64 as f64 / count;
        }
        println!("avg cost: {avg_cost}");
        println!("accuracy: {accuracy}%");
    }

    Ok(())
}

fn expected(n: u8) -> network::Vector {
    let mut vector = network::Vector::from_element(10, 0.0);
    if (0..10).contains(&n) {
        vector[n as usize] = 1.0;
    }
    vector
}

fn print_info(label: u8, image: &loader::Image, nn: &network::Network) {
    println!("{}", label);
    println!("{}", image);
    println!(
        " {}",
        (0..10).map(|n| format!("{:<3}", n)).collect::<String>()
    );
    println!(
        "{}",
        nn.output()
            .iter()
            .copied()
            .map(|n| {
                format!(
                    "\x1b[48;2;0;0;0m\x1b[38;2;{px};{px};{px}m ● \x1b[0m",
                    px = (n * (0xff as f64)) as u8
                )
            })
            .collect::<String>()
    );
    println!(
        "{}",
        nn.output()
            .iter()
            .copied()
            .map(|n| (n * 100.0).round() as u8)
            .map(|n| format!("{:<3}", n))
            .collect::<String>()
    );
    println!(
        "{:?}",
        nn.output()
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    );
    println!("cost: {}", nn.cost(&expected(label)));
}

fn save(conf: &network::NetConf) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = conf
        .flatten()
        .iter()
        .copied()
        .flat_map(|n| n.to_be_bytes().into_iter())
        .collect::<Vec<_>>();
    let mut file = std::fs::File::create("network")?;
    file.write_all(&bytes)?;
    Ok(())
}

#[allow(dead_code)]
fn gradient_check(label: u8, image: &loader::Image, nn: &mut network::Network) {
    const VARIANCE: f64 = 1.0e-10;

    let expected = expected(label);
    let input_vector = (*image).into();
    nn.process(&input_vector);
    let gradient = nn.gradient(&expected);
    let mut actual = network::Vector::from_element(gradient.nrows(), 0.0);
    let mut flattened = nn.conf.read().unwrap().flatten();
    for i in 0..flattened.nrows() {
        flattened[i] += VARIANCE;
        nn.conf
            .write()
            .unwrap()
            .load_iter(flattened.iter().copied());
        nn.process(&input_vector);
        let upper = nn.cost(&expected);
        flattened[i] -= 2.0 * VARIANCE;
        nn.conf
            .write()
            .unwrap()
            .load_iter(flattened.iter().copied());
        nn.process(&input_vector);
        let lower = nn.cost(&expected);
        flattened[i] += VARIANCE;
        actual[i] = (upper - lower) / (2.0 * VARIANCE);
    }

    let mut diff = &actual - &gradient;
    println!("{}", diff.iter().copied().any(f64::is_nan));
    diff.apply(|n| *n = n.abs());
    println!("{}", diff.iter().copied().any(f64::is_nan));
    (0..diff.nrows()).for_each(|i| {
        diff[i] /= actual[i].abs().max(gradient[i].abs()) + VARIANCE;
    });
    println!("{}", diff.iter().copied().any(f64::is_nan));

    let sum = diff.sum();
    let avg = sum / diff.nrows() as f64;
    let max = diff.max();
    println!("{}", diff.iter().copied().any(f64::is_nan));
    println!("avg: {avg}, sum: {sum}, max: {max}");
}

fn gen_bar(n: usize) -> String {
    let head = match n % 8 {
        0 => ' ',
        1 => '\u{258f}',
        2 => '\u{258e}',
        3 => '\u{258d}',
        4 => '\u{258c}',
        5 => '\u{258b}',
        6 => '\u{258a}',
        7 => '\u{2589}',
        _ => unreachable!(),
    };
    let mut bar = "\u{2588}".repeat(n / 8);
    bar.push(head);
    bar
}
