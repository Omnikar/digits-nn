pub fn load_config() -> anyhow::Result<Config> {
    use std::io::Read;

    let mut f = std::fs::File::open("config.ron")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;

    ron::from_str(&s).map_err(Into::into)
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub data: Datasets,
    pub h_layers: Vec<usize>,
    pub learning_rate: f64,
    pub momentum_decay: f64,
    pub batch_size: usize,
    pub epochs: usize,
}

#[derive(serde::Deserialize)]
pub struct Datasets {
    pub train: Dataset,
    pub test: Dataset,
}

#[derive(serde::Deserialize)]
pub struct Dataset {
    pub labels: String,
    pub images: String,
}
