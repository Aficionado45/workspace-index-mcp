use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use crate::model::ErrorCode;

const MAX_CURSOR_BYTES: usize = 64 * 1024;
type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPayload {
    pub context_digest: String,
    pub state_digest: String,
    pub offset: usize,
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = value.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let mut output = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        output.push((digit(pair[0])? << 4) | digit(pair[1])?);
    }
    Some(output)
}

pub fn generate_cursor(secret: &[u8], payload: &CursorPayload) -> String {
    let json = serde_json::to_string(payload).unwrap();
    let mut mac = HmacSha256::new_from_slice(secret).unwrap();
    mac.update(json.as_bytes());
    let signature = mac.finalize().into_bytes();
    
    let sig_hex = encode_hex(&signature);
    let json_hex = encode_hex(json.as_bytes());
    
    format!("{}.{}", json_hex, sig_hex)
}

pub fn verify_cursor(secret: &[u8], cursor_str: &str) -> Result<CursorPayload, ErrorCode> {
    if cursor_str.len() > MAX_CURSOR_BYTES {
        return Err(ErrorCode::StaleCursor);
    }
    let (json_hex, signature_hex) = cursor_str.split_once('.').ok_or(ErrorCode::StaleCursor)?;
    if signature_hex.contains('.') {
        return Err(ErrorCode::StaleCursor);
    }
    let json_bytes = decode_hex(json_hex).ok_or(ErrorCode::StaleCursor)?;
    let signature = decode_hex(signature_hex).ok_or(ErrorCode::StaleCursor)?;

    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| ErrorCode::StaleCursor)?;
    mac.update(&json_bytes);
    mac.verify_slice(&signature).map_err(|_| ErrorCode::StaleCursor)?;

    serde_json::from_slice(&json_bytes).map_err(|_| ErrorCode::StaleCursor)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_multibyte_hex_at_different_byte_offsets() {
        for cursor in ["aé000.00", "000é0.00", "00.00aé0"] {
            assert_eq!(
                verify_cursor(b"secret", cursor).unwrap_err(),
                ErrorCode::StaleCursor,
            );
        }
    }

    #[test]
    fn rejects_oversized_cursor_before_decoding() {
        let cursor = "0".repeat(MAX_CURSOR_BYTES + 1);
        assert_eq!(
            verify_cursor(b"secret", &cursor).unwrap_err(),
            ErrorCode::StaleCursor,
        );
    }
}
