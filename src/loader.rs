use crate::IMAGE_SIZE;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const LABELS_MNUM: u32 = 0x00000801;
const IMAGES_MNUM: u32 = 0x00000803;

#[derive(Clone, Copy)]
pub struct Image {
    pub pixels: [u8; IMAGE_SIZE.pow(2)],
}

impl std::fmt::Display for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self
            .pixels
            .chunks(IMAGE_SIZE)
            .map(|row| {
                row.iter()
                    .copied()
                    .map(|px| format!("\x1b[48;2;{px};{px};{px}m  \x1b[0m"))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{}", s)
    }
}

impl From<Image> for crate::network::Vector {
    fn from(img: Image) -> Self {
        // Not using `Vector::from_iterator` because of a rust-analyzer bug
        Self::from_column_slice(
            &img.pixels
                .iter()
                .map(|n| *n as f64 / 0xff as f64)
                .collect::<Vec<_>>(),
        )
    }
}

pub fn load_labels(path: impl AsRef<Path>) -> anyhow::Result<Vec<u8>> {
    let mut f = File::open(path)?;

    let mut read_u32 = || -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    };

    let mnum = read_u32()?;
    if mnum != LABELS_MNUM {
        anyhow::bail!("invalid labels mnum: {mnum} (expected {LABELS_MNUM})");
    }

    let len = read_u32()? as usize;

    let mut label_bytes = Vec::with_capacity(len);
    f.read_to_end(&mut label_bytes)?;

    Ok(label_bytes)
}

pub fn load_images(path: impl AsRef<Path>) -> anyhow::Result<Vec<Image>> {
    let mut f = File::open(path)?;

    let mut read_u32 = || -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    };

    let mnum = read_u32()?;
    if mnum != IMAGES_MNUM {
        anyhow::bail!("invalid images mnum: {mnum} (expected {IMAGES_MNUM})");
    }

    let len = read_u32()? as usize;

    let width = read_u32()? as usize;
    let height = read_u32()? as usize;
    if !(width == IMAGE_SIZE && height == IMAGE_SIZE) {
        anyhow::bail!(
            "invalid image size: {width}x{height} (expected {0}x{0})",
            IMAGE_SIZE
        );
    }

    let mut image_bytes = Vec::with_capacity(len);
    f.read_to_end(&mut image_bytes)?;
    let images = image_bytes
        .chunks(IMAGE_SIZE.pow(2))
        .map(|chunk| {
            let mut arr = [0u8; IMAGE_SIZE.pow(2)];
            arr.copy_from_slice(chunk);
            Image { pixels: arr }
        })
        .collect();
    Ok(images)
}
