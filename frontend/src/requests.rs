use polars::io::SerWriter;
use polars::prelude::{CsvWriter, DataFrame};
use reqwest::blocking::{Client, multipart::{Form, Part}};

/// Blocking version: no async/Tokio required.
pub fn request_bar_chart_blocking(
    df: &mut DataFrame,
    key: String,
    value: String,
) -> Result<String, reqwest::Error> {
    // 1) Serialize the DataFrame as CSV into a Vec<u8> buffer:
    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .has_header(true)
        .finish(df)
        .unwrap();

    // 2) Build each multipart part:
    let csv_part = Part::bytes(buf)
        .file_name("data.csv")
        .mime_str("text/csv")
        .unwrap();
    let key_part = Part::text(key);
    let value_part = Part::text(value);

    // 3) Assemble into a Form:
    let form = Form::new()
        .part("file", csv_part)
        .part("key", key_part)
        .part("value", value_part);

    // 4) Use a blocking Client to POST:
    let client = Client::new();
    let resp = client
        .post("http://127.0.0.1:8000/upload_csv_with_keys")
        .multipart(form)
        .send()?;      // <— blocking send()
    let resp_text = resp.text()?; // <— blocking text()

    Ok(resp_text)
}