use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSummary {
    pub path: String,
    pub file_name: String,
    pub kind: String,
    pub entries: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArrayData {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub total_elements: usize,
    pub values: Vec<String>,
    pub field_names: Vec<String>,
    pub components: Vec<String>,
}

#[derive(Debug)]
struct NpyMeta {
    descr: String,
    shape: Vec<usize>,
    data_offset: usize,
    fortran_order: bool,
}

#[derive(Clone, Copy, Debug)]
enum Endian {
    Little,
    Big,
    Native,
    None,
}

#[derive(Clone, Copy, Debug)]
struct Descriptor {
    endian: Endian,
    kind: char,
    size: usize,
}

pub fn inspect_file(path: &Path) -> Result<FileSummary, String> {
    if !path.is_file() {
        return Err(format!("File does not exist: {}", path.display()));
    }
    let extension = extension(path)?;
    let path_text = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("NumPy file")
        .to_string();

    let entries = match extension.as_str() {
        "npy" => vec![file_name.clone()],
        "npz" => list_npz_entries(path)?,
        _ => return Err("Select a .npy or .npz file".to_string()),
    };

    Ok(FileSummary {
        path: path_text,
        file_name,
        kind: extension,
        entries,
    })
}

pub fn load_array(path: &Path, entry: Option<&str>) -> Result<ArrayData, String> {
    if !path.is_file() {
        return Err(format!("File does not exist: {}", path.display()));
    }

    let extension = extension(path)?;
    let (bytes, name) = match extension.as_str() {
        "npy" => (
            std::fs::read(path).map_err(|err| format!("Cannot read {}: {err}", path.display()))?,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("array.npy")
                .to_string(),
        ),
        "npz" => {
            let entry = entry.ok_or_else(|| "Choose an array in the NPZ file".to_string())?;
            (read_npz_entry(path, entry)?, entry.to_string())
        }
        _ => return Err("Select a .npy or .npz file".to_string()),
    };

    decode_npy(&bytes, name)
}

fn extension(path: &Path) -> Result<String, String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| "File has no extension".to_string())
}

fn list_npz_entries(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path).map_err(|err| format!("Cannot open {}: {err}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err| format!("Invalid NPZ file {}: {err}", path.display()))?;
    let mut names = Vec::new();
    for index in 0..archive.len() {
        let item = archive
            .by_index(index)
            .map_err(|err| format!("Cannot read NPZ entry: {err}"))?;
        if !item.is_dir() && item.name().to_ascii_lowercase().ends_with(".npy") {
            names.push(item.name().trim_end_matches(".npy").to_string());
        }
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Err("The NPZ archive contains no .npy arrays".to_string());
    }
    Ok(names)
}

fn read_npz_entry(path: &Path, entry: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|err| format!("Cannot open {}: {err}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err| format!("Invalid NPZ file {}: {err}", path.display()))?;
    let raw_name = if entry.to_ascii_lowercase().ends_with(".npy") {
        entry.to_string()
    } else {
        format!("{entry}.npy")
    };
    let mut item = archive
        .by_name(&raw_name)
        .map_err(|err| format!("Cannot open array '{entry}': {err}"))?;
    let mut bytes = Vec::with_capacity(item.size().min(usize::MAX as u64) as usize);
    item.read_to_end(&mut bytes)
        .map_err(|err| format!("Cannot read array '{entry}': {err}"))?;
    Ok(bytes)
}

fn decode_npy(bytes: &[u8], name: String) -> Result<ArrayData, String> {
    let meta = parse_npy_meta(bytes)?;

    if meta.descr.trim_start().starts_with('[') {
        return decode_structured(bytes, name, meta);
    }

    let descriptor = parse_descriptor(&meta.descr)
        .ok_or_else(|| format!("Unsupported dtype descriptor '{}'", meta.descr))?;
    if matches!(descriptor.kind, 'U' | 'S') {
        return decode_string_array(bytes, name, meta, descriptor);
    }
    if descriptor.kind == 'O'
        || matches!(
            (descriptor.kind, descriptor.size),
            ('f', 2) | ('M' | 'm', 8) | ('V', _)
        )
    {
        return decode_special_array(bytes, name, meta, descriptor);
    }

    decode_numeric_array(bytes, name, meta, descriptor)
}

fn decode_string_array(
    bytes: &[u8],
    name: String,
    meta: NpyMeta,
    descriptor: Descriptor,
) -> Result<ArrayData, String> {
    let count = element_count(&meta.shape)?;
    let item_bytes = if descriptor.kind == 'U' {
        descriptor
            .size
            .checked_mul(4)
            .ok_or_else(|| "dtype size overflow".to_string())?
    } else {
        descriptor.size
    };
    let required = count
        .checked_mul(item_bytes)
        .and_then(|size| meta.data_offset.checked_add(size))
        .ok_or_else(|| "array size overflow".to_string())?;
    if required > bytes.len() {
        return Err("NPY data is shorter than its header declares".to_string());
    }

    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let start = meta.data_offset + index * item_bytes;
        let value = if descriptor.kind == 'U' {
            decode_unicode(&bytes[start..start + item_bytes], descriptor.endian)?
        } else {
            String::from_utf8_lossy(&bytes[start..start + item_bytes])
                .trim_end_matches('\0')
                .to_string()
        };
        values.push(value);
    }

    Ok(ArrayData {
        name,
        dtype: meta.descr,
        shape: meta.shape,
        total_elements: count,
        values,
        field_names: Vec::new(),
        components: Vec::new(),
    })
}

fn decode_structured(bytes: &[u8], name: String, meta: NpyMeta) -> Result<ArrayData, String> {
    let fields = parse_structured_fields(&meta.descr)?;
    if fields.is_empty() {
        return Err("Structured dtype has no fields".to_string());
    }

    let mut parsed = Vec::with_capacity(fields.len());
    let mut record_bytes = 0usize;
    for (field_name, field_dtype) in &fields {
        let descriptor = parse_descriptor(field_dtype)
            .ok_or_else(|| format!("Unsupported field dtype '{field_dtype}'"))?;
        let bytes = descriptor_bytes(descriptor)?;
        parsed.push((field_name.clone(), descriptor, record_bytes));
        record_bytes = record_bytes
            .checked_add(bytes)
            .ok_or_else(|| "dtype size overflow".to_string())?;
    }

    let records = element_count(&meta.shape)?;
    let required = records
        .checked_mul(record_bytes)
        .and_then(|size| meta.data_offset.checked_add(size))
        .ok_or_else(|| "array size overflow".to_string())?;
    if required > bytes.len() {
        return Err("NPY structured data is shorter than its header declares".to_string());
    }

    let mut values = Vec::with_capacity(records.saturating_mul(fields.len()));
    for row in 0..records {
        let base = meta.data_offset + row * record_bytes;
        for (_, descriptor, offset) in &parsed {
            let size = descriptor_bytes(*descriptor)?;
            values.push(decode_scalar(
                &bytes[base + offset..base + offset + size],
                *descriptor,
            )?);
        }
    }

    let mut shape = meta.shape;
    if shape.is_empty() {
        shape.push(1);
    }
    shape.push(fields.len());
    Ok(ArrayData {
        name,
        dtype: meta.descr,
        shape,
        total_elements: records.saturating_mul(fields.len()),
        values,
        field_names: fields.into_iter().map(|(name, _)| name).collect(),
        components: Vec::new(),
    })
}

fn parse_npy_meta(bytes: &[u8]) -> Result<NpyMeta, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("Not a valid NPY file".to_string());
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("Truncated NPY header".to_string());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        _ => return Err(format!("Unsupported NPY version {major}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header length overflow".to_string())?;
    if header_end > bytes.len() {
        return Err("Truncated NPY header".to_string());
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| "NPY header is not valid UTF-8/ASCII".to_string())?;

    Ok(NpyMeta {
        descr: extract_descr(header)?,
        shape: extract_shape(header)?,
        data_offset: header_end,
        fortran_order: extract_fortran_order(header)?,
    })
}

fn extract_descr(header: &str) -> Result<String, String> {
    let key = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| "NPY header has no descr field".to_string())?;
    let colon = header[key..]
        .find(':')
        .map(|value| key + value + 1)
        .ok_or_else(|| "Invalid descr field".to_string())?;
    let rest = header[colon..].trim_start();
    let first = rest
        .chars()
        .next()
        .ok_or_else(|| "Empty descr field".to_string())?;
    if first == '\'' || first == '"' {
        let end = rest[1..]
            .find(first)
            .ok_or_else(|| "Unterminated descr string".to_string())?;
        return Ok(rest[1..1 + end].to_string());
    }
    if first == '[' {
        let end = rest
            .find(']')
            .ok_or_else(|| "Unterminated structured descr".to_string())?;
        return Ok(rest[..=end].to_string());
    }
    Err("Unsupported descr syntax".to_string())
}

fn extract_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| "NPY header has no shape field".to_string())?;
    let colon = header[key..]
        .find(':')
        .map(|value| key + value + 1)
        .ok_or_else(|| "Invalid shape field".to_string())?;
    let open = header[colon..]
        .find('(')
        .map(|value| colon + value + 1)
        .ok_or_else(|| "Invalid shape tuple".to_string())?;
    let close = header[open..]
        .find(')')
        .map(|value| open + value)
        .ok_or_else(|| "Unterminated shape tuple".to_string())?;
    let raw = header[open..close].trim();
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("Invalid shape dimension '{part}'"))
        })
        .collect()
}

fn extract_fortran_order(header: &str) -> Result<bool, String> {
    let key = header
        .find("'fortran_order'")
        .or_else(|| header.find("\"fortran_order\""))
        .ok_or_else(|| "NPY header has no fortran_order field".to_string())?;
    let colon = header[key..]
        .find(':')
        .map(|value| key + value + 1)
        .ok_or_else(|| "Invalid fortran_order field".to_string())?;
    let value = header[colon..].trim_start();
    if value.starts_with("False") {
        Ok(false)
    } else if value.starts_with("True") {
        Ok(true)
    } else {
        Err("Invalid fortran_order value".to_string())
    }
}

fn parse_descriptor(raw: &str) -> Option<Descriptor> {
    let mut chars = raw.trim().chars();
    let first = chars.next()?;
    let (endian, kind) = match first {
        '<' => (Endian::Little, chars.next()?),
        '>' => (Endian::Big, chars.next()?),
        '=' => (Endian::Native, chars.next()?),
        '|' => (Endian::None, chars.next()?),
        kind => (Endian::Native, kind),
    };
    let size_text: String = chars.take_while(|ch| ch.is_ascii_digit()).collect();
    let size = if kind == '?' {
        1
    } else if kind == 'O' && size_text.is_empty() {
        0
    } else {
        size_text.parse().ok()?
    };
    Some(Descriptor { endian, kind, size })
}

fn parse_structured_fields(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut cursor = raw
        .trim()
        .strip_prefix('[')
        .ok_or_else(|| "Invalid structured dtype".to_string())?;
    let mut fields = Vec::new();
    loop {
        cursor = cursor.trim_start_matches(|ch: char| ch.is_whitespace() || ch == ',');
        if cursor.starts_with(']') {
            break;
        }
        cursor = cursor
            .strip_prefix('(')
            .ok_or_else(|| "Invalid structured field tuple".to_string())?
            .trim_start();
        let (name, rest) = take_quoted(cursor)?;
        cursor = rest.trim_start();
        cursor = cursor
            .strip_prefix(',')
            .ok_or_else(|| "Invalid structured field separator".to_string())?
            .trim_start();
        let (dtype, rest) = take_quoted(cursor)?;
        fields.push((name, dtype));
        let close = rest
            .find(')')
            .ok_or_else(|| "Unterminated structured field tuple".to_string())?;
        cursor = &rest[close + 1..];
    }
    Ok(fields)
}

fn take_quoted(raw: &str) -> Result<(String, &str), String> {
    let quote = raw
        .chars()
        .next()
        .ok_or_else(|| "Missing quoted value".to_string())?;
    if quote != '\'' && quote != '"' {
        return Err("Expected quoted value".to_string());
    }
    let end = raw[1..]
        .find(quote)
        .ok_or_else(|| "Unterminated quoted value".to_string())?;
    Ok((raw[1..1 + end].to_string(), &raw[2 + end..]))
}

fn descriptor_bytes(descriptor: Descriptor) -> Result<usize, String> {
    match descriptor.kind {
        'U' => descriptor
            .size
            .checked_mul(4)
            .ok_or_else(|| "dtype size overflow".to_string()),
        '?' | 'b' | 'i' | 'u' | 'f' | 'c' | 'S' => Ok(descriptor.size),
        other => Err(format!("Unsupported field dtype kind '{other}'")),
    }
}

macro_rules! read_num {
    ($bytes:expr, $type:ty, $little:expr) => {{
        let raw: [u8; std::mem::size_of::<$type>()] = $bytes
            .get(..std::mem::size_of::<$type>())
            .ok_or_else(|| "Truncated scalar value".to_string())?
            .try_into()
            .map_err(|_| "Truncated scalar value".to_string())?;
        if $little {
            <$type>::from_le_bytes(raw)
        } else {
            <$type>::from_be_bytes(raw)
        }
    }};
}

fn decode_numeric_array(
    bytes: &[u8],
    name: String,
    meta: NpyMeta,
    descriptor: Descriptor,
) -> Result<ArrayData, String> {
    if !matches!(
        (descriptor.kind, descriptor.size),
        ('?', 1)
            | ('b', 1)
            | ('i', 1 | 2 | 4 | 8)
            | ('u', 1 | 2 | 4 | 8)
            | ('f', 4 | 8)
            | ('c', 8 | 16)
    ) {
        return Err(format!(
            "Unsupported dtype '{}'. Supported: bool, integers, f32/f64, complex64/128, byte/unicode strings, and scalar structured fields",
            meta.descr
        ));
    }

    let count = element_count(&meta.shape)?;
    let required = count
        .checked_mul(descriptor.size)
        .and_then(|size| meta.data_offset.checked_add(size))
        .ok_or_else(|| "array size overflow".to_string())?;
    if required > bytes.len() {
        return Err("NPY data is shorter than its header declares".to_string());
    }

    let fortran_strides = if meta.fortran_order {
        let mut stride = 1usize;
        let mut strides = Vec::with_capacity(meta.shape.len());
        for dimension in &meta.shape {
            strides.push(stride);
            stride *= *dimension;
        }
        Some(strides)
    } else {
        None
    };
    let complex = descriptor.kind == 'c';
    let capacity = if complex {
        count.saturating_mul(2)
    } else {
        count
    };
    let mut values = Vec::with_capacity(capacity);
    let little = matches!(descriptor.endian, Endian::Little | Endian::None)
        || (matches!(descriptor.endian, Endian::Native) && cfg!(target_endian = "little"));

    for logical_index in 0..count {
        let source_index = if let Some(strides) = &fortran_strides {
            let mut remainder = logical_index;
            let mut index = 0usize;
            for axis in (0..meta.shape.len()).rev() {
                let coordinate = remainder % meta.shape[axis];
                remainder /= meta.shape[axis];
                index += coordinate * strides[axis];
            }
            index
        } else {
            logical_index
        };
        let start = meta.data_offset + source_index * descriptor.size;
        let chunk = &bytes[start..start + descriptor.size];
        match (descriptor.kind, descriptor.size) {
            ('c', 8) => {
                values.push(read_num!(&chunk[..4], f32, little).to_string());
                values.push(read_num!(&chunk[4..], f32, little).to_string());
            }
            ('c', 16) => {
                values.push(read_num!(&chunk[..8], f64, little).to_string());
                values.push(read_num!(&chunk[8..], f64, little).to_string());
            }
            _ => values.push(decode_scalar(chunk, descriptor)?),
        }
    }

    Ok(ArrayData {
        name,
        dtype: meta.descr,
        shape: meta.shape,
        total_elements: count,
        values,
        field_names: Vec::new(),
        components: if complex {
            vec!["real".to_string(), "imag".to_string()]
        } else {
            Vec::new()
        },
    })
}

fn decode_scalar(bytes: &[u8], descriptor: Descriptor) -> Result<String, String> {
    let little = matches!(descriptor.endian, Endian::Little | Endian::None)
        || (matches!(descriptor.endian, Endian::Native) && cfg!(target_endian = "little"));
    let value = match (descriptor.kind, descriptor.size) {
        ('?', 1) | ('b', 1) => (bytes[0] != 0).to_string(),
        ('i', 1) => (bytes[0] as i8).to_string(),
        ('u', 1) => bytes[0].to_string(),
        ('i', 2) => read_num!(bytes, i16, little).to_string(),
        ('i', 4) => read_num!(bytes, i32, little).to_string(),
        ('i', 8) => read_num!(bytes, i64, little).to_string(),
        ('u', 2) => read_num!(bytes, u16, little).to_string(),
        ('u', 4) => read_num!(bytes, u32, little).to_string(),
        ('u', 8) => read_num!(bytes, u64, little).to_string(),
        ('f', 4) => read_num!(bytes, f32, little).to_string(),
        ('f', 8) => read_num!(bytes, f64, little).to_string(),
        ('c', 8) => {
            let re = read_num!(&bytes[..4], f32, little);
            let im = read_num!(&bytes[4..], f32, little);
            format!("{re}{:+}j", im)
        }
        ('c', 16) => {
            let re = read_num!(&bytes[..8], f64, little);
            let im = read_num!(&bytes[8..], f64, little);
            format!("{re}{:+}j", im)
        }
        ('S', _) => String::from_utf8_lossy(bytes)
            .trim_end_matches('\0')
            .to_string(),
        ('U', _) => decode_unicode(bytes, descriptor.endian)?,
        _ => {
            return Err(format!(
                "Unsupported scalar dtype '{}{}'",
                descriptor.kind, descriptor.size
            ))
        }
    };
    Ok(value)
}

fn decode_unicode(bytes: &[u8], endian: Endian) -> Result<String, String> {
    if bytes.len() % 4 != 0 {
        return Err("Invalid Unicode dtype width".to_string());
    }
    let little = matches!(endian, Endian::Little | Endian::None)
        || (matches!(endian, Endian::Native) && cfg!(target_endian = "little"));
    let mut value = String::new();
    for chunk in bytes.chunks_exact(4) {
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| "Invalid Unicode value".to_string())?;
        let codepoint = if little {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        };
        if codepoint == 0 {
            break;
        }
        value.push(char::from_u32(codepoint).unwrap_or('\u{fffd}'));
    }
    Ok(value)
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "array element count overflow".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::WriteNpyExt;

    #[test]
    fn parses_v1_header() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }\n";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        let meta = parse_npy_meta(&bytes).unwrap();
        assert_eq!(meta.descr, "<f8");
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.data_offset, bytes.len());
    }

    #[test]
    fn decodes_numeric_matrix_written_by_ndarray_npy() {
        let array = ndarray::array![[1.25_f64, -2.0], [3.5, 4.0]];
        let mut bytes = Vec::new();
        array.write_npy(&mut bytes).unwrap();

        let decoded = decode_npy(&bytes, "matrix".to_string()).unwrap();
        assert_eq!(decoded.shape, vec![2, 2]);
        assert_eq!(decoded.total_elements, 4);
        assert_eq!(decoded.values, vec!["1.25", "-2", "3.5", "4"]);
    }

    #[test]
    fn decodes_unicode_array() {
        let header = "{'descr': '<U2', 'fortran_order': False, 'shape': (2,), }\n";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for codepoint in ['A' as u32, '\u{03b2}' as u32, '\u{732b}' as u32, 0] {
            bytes.extend_from_slice(&codepoint.to_le_bytes());
        }

        let decoded = decode_npy(&bytes, "words".to_string()).unwrap();
        assert_eq!(decoded.shape, vec![2]);
        assert_eq!(decoded.values, vec!["A\u{03b2}", "\u{732b}"]);
    }

    #[test]
    fn parses_structured_fields() {
        let fields = parse_structured_fields("[('time', '<f8'), ('flag', '|u1')]").unwrap();
        assert_eq!(
            fields,
            vec![
                ("time".to_string(), "<f8".to_string()),
                ("flag".to_string(), "|u1".to_string())
            ]
        );
    }
}

fn decode_special_array(
    bytes: &[u8],
    name: String,
    meta: NpyMeta,
    descriptor: Descriptor,
) -> Result<ArrayData, String> {
    let count = element_count(&meta.shape)?;
    if descriptor.kind == 'O' {
        return Ok(ArrayData {
            name,
            dtype: meta.descr,
            shape: meta.shape,
            total_elements: count,
            values: vec!["<Python object: pickle not loaded>".to_string(); count],
            field_names: Vec::new(),
            components: Vec::new(),
        });
    }

    let item_bytes = descriptor.size;
    if item_bytes == 0 {
        return Err(format!("Invalid zero-width dtype '{}'", meta.descr));
    }
    let required = count
        .checked_mul(item_bytes)
        .and_then(|size| meta.data_offset.checked_add(size))
        .ok_or_else(|| "array size overflow".to_string())?;
    if required > bytes.len() {
        return Err("NPY data is shorter than its header declares".to_string());
    }

    let little = matches!(descriptor.endian, Endian::Little | Endian::None)
        || (matches!(descriptor.endian, Endian::Native) && cfg!(target_endian = "little"));
    let unit = meta
        .descr
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(unit, _)| unit)
        .unwrap_or("ticks");
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let start = meta.data_offset + index * item_bytes;
        let chunk = &bytes[start..start + item_bytes];
        let text = match (descriptor.kind, descriptor.size) {
            ('f', 2) => {
                let raw = if little {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                };
                half_to_f32(raw).to_string()
            }
            ('M' | 'm', 8) => {
                let raw: [u8; 8] = chunk
                    .try_into()
                    .map_err(|_| "Truncated datetime value".to_string())?;
                let value = if little {
                    i64::from_le_bytes(raw)
                } else {
                    i64::from_be_bytes(raw)
                };
                if value == i64::MIN {
                    "NaT".to_string()
                } else {
                    format!("{value} {unit}")
                }
            }
            ('V', _) => {
                let mut output = String::with_capacity(chunk.len() * 2 + 2);
                output.push_str("0x");
                for byte in chunk {
                    use std::fmt::Write;
                    let _ = write!(output, "{byte:02x}");
                }
                output
            }
            _ => return Err(format!("Unsupported special dtype '{}'", meta.descr)),
        };
        values.push(text);
    }

    Ok(ArrayData {
        name,
        dtype: meta.descr,
        shape: meta.shape,
        total_elements: count,
        values,
        field_names: Vec::new(),
        components: Vec::new(),
    })
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    let output = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut mantissa = fraction;
                let mut shift = 0i32;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    shift += 1;
                }
                mantissa &= 0x03ff;
                let exponent32 = (127 - 15 - shift + 1) as u32;
                sign | (exponent32 << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (((exponent + 127 - 15) as u32) << 23) | (fraction << 13),
    };
    f32::from_bits(output)
}

#[cfg(test)]
mod special_dtype_tests {
    use super::*;

    fn npy_bytes(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        npy_bytes_with_order(descr, shape, false, data)
    }

    fn npy_bytes_with_order(descr: &str, shape: &str, fortran_order: bool, data: &[u8]) -> Vec<u8> {
        let order = if fortran_order { "True" } else { "False" };
        let header =
            format!("{{'descr': '{descr}', 'fortran_order': {order}, 'shape': ({shape}), }}\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn decodes_float16() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x3e00u16.to_le_bytes());
        data.extend_from_slice(&0xc000u16.to_le_bytes());
        let decoded = decode_npy(&npy_bytes("<f2", "2,", &data), "half".to_string()).unwrap();
        assert_eq!(decoded.values, vec!["1.5", "-2"]);
    }

    #[test]
    fn decodes_big_endian_integers() {
        let mut data = Vec::new();
        data.extend_from_slice(&(-300_i16).to_be_bytes());
        data.extend_from_slice(&1200_i16.to_be_bytes());
        let decoded = decode_npy(&npy_bytes(">i2", "2,", &data), "big-endian".to_string()).unwrap();
        assert_eq!(decoded.values, vec!["-300", "1200"]);
    }

    #[test]
    fn splits_complex_values_into_components() {
        let mut data = Vec::new();
        for value in [1.25_f32, -2.5, 3.0, 4.5] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let decoded = decode_npy(&npy_bytes("<c8", "2,", &data), "complex".to_string()).unwrap();
        assert_eq!(decoded.values, vec!["1.25", "-2.5", "3", "4.5"]);
        assert_eq!(decoded.components, vec!["real", "imag"]);
    }

    #[test]
    fn normalizes_fortran_order_to_row_major() {
        let mut data = Vec::new();
        for value in [1_i32, 4, 2, 5, 3, 6] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let decoded = decode_npy(
            &npy_bytes_with_order("<i4", "2, 3", true, &data),
            "fortran".to_string(),
        )
        .unwrap();
        assert_eq!(decoded.shape, vec![2, 3]);
        assert_eq!(decoded.values, vec!["1", "2", "3", "4", "5", "6"]);
    }

    #[test]
    fn reports_object_array_without_unpickling() {
        let decoded = decode_npy(
            &npy_bytes("|O", "2,", b"pickle payload is deliberately ignored"),
            "objects".to_string(),
        )
        .unwrap();
        assert_eq!(decoded.shape, vec![2]);
        assert!(decoded.values[0].contains("pickle not loaded"));
    }
}
