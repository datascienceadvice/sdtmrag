use sdtmrag::config::Config;
use sdtmrag::embed::Embedder;
use std::time::Instant;

/// Быстрая проверка пайплайна: один эмбеддинг одного предложения.
/// `cargo run --release --example smoke`
fn main() -> anyhow::Result<()> {
    let cfg = Config::load(None)?;
    println!(
        "model: {} | cache: {} | threads: {}",
        cfg.embedding.model,
        cfg.models_dir.display(),
        cfg.embedding.intra_threads
    );

    let t0 = Instant::now();
    let mut embedder = Embedder::from_config(&cfg)?;
    println!("model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let sentence = "USUBJID uniquely identifies a subject in a study and is a required variable in every SDTM domain.".to_string();

    let t1 = Instant::now();
    let v = embedder.embed_query(&sentence)?;
    println!(
        "embedded 1 sentence in {:.0}ms | dim={} | norm={:.4} | head={:?}",
        t1.elapsed().as_millis(),
        v.len(),
        v.iter().map(|x| x * x).sum::<f32>().sqrt(),
        &v[..3]
    );

    let n = 100;
    let texts = vec![sentence.clone(); n];
    let t2 = Instant::now();
    let batch = embedder.embed_documents(&texts)?;
    let dt = t2.elapsed();
    println!(
        "embedded {n} sentences in {:.2}s => {:.1} ch/s ({:.1} ms/ch)",
        dt.as_secs_f64(),
        n as f64 / dt.as_secs_f64(),
        dt.as_millis() as f64 / n as f64
    );
    assert_eq!(batch.len(), n);
    println!("smoke OK");
    Ok(())
}