use polars::io::SerWriter;
use polars::prelude::{CsvWriter, DataFrame};
use reqwest::blocking::{Client, multipart::{Form, Part}};

pub fn request_bar_chart_blocking(
    df: &mut DataFrame,
    key: String,
    value: String,
) -> Result<String, reqwest::Error> {
    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .has_header(true)
        .finish(df)
        .unwrap();

    let csv_part = Part::bytes(buf)
        .file_name("data.csv")
        .mime_str("text/csv")
        .unwrap();
    let key_part = Part::text(key);
    let value_part = Part::text(value);

    let form = Form::new()
        .part("file", csv_part)
        .part("key", key_part)
        .part("value", value_part);

    let client = Client::new();
    let resp = client
        .post("http://127.0.0.1:8000/upload_csv_with_keys")
        .multipart(form)
        .send()?;
    let resp_text = resp.text()?;

    Ok(resp_text)
}
