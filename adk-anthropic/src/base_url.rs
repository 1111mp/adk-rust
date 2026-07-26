//! Base URL validation shared by every client in this crate.
//!
//! Each client attaches the Anthropic API key to every outbound request, so a
//! base URL that is not encrypted would transmit the credential in cleartext.
//! The validator lives here — outside any feature gate — so the default
//! [`crate::Anthropic`] client and the feature-gated Managed Agents client
//! enforce exactly the same rule.

use crate::error::{Error, Result};

/// Reject base URLs that would transmit the API key in cleartext.
///
/// Accepts `https://` unconditionally and `http://` only when the host is
/// loopback (`localhost`, `127.0.0.0/8`, `[::1]`) for local development.
/// Everything else — other schemes, unparseable input — is a validation error
/// naming the offending scheme.
pub(crate) fn validate_base_url(base_url: &str) -> Result<()> {
    let parsed = url::Url::parse(base_url).map_err(|e| {
        Error::validation(
            format!(
                "base URL '{base_url}' is not a valid absolute URL ({e}). \
                 Provide a full URL such as https://api.anthropic.com."
            ),
            Some("base_url".to_string()),
        )
    })?;

    if parsed.scheme().eq_ignore_ascii_case("https") {
        return Ok(());
    }

    if parsed.scheme().eq_ignore_ascii_case("http") && is_loopback_host(&parsed) {
        return Ok(());
    }

    Err(Error::validation(
        format!(
            "base URL '{base_url}' uses scheme '{}', which would send the Anthropic API key \
             over an unencrypted connection. Use https://, or http:// with a loopback host \
             (localhost, 127.0.0.1, [::1]) for local development.",
            parsed.scheme()
        ),
        Some("base_url".to_string()),
    ))
}

/// True when the URL host is a loopback address or `localhost`.
fn is_loopback_host(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_base_url_accepts_https() {
        assert!(validate_base_url("https://api.anthropic.com").is_ok());
    }

    #[test]
    fn test_validate_base_url_accepts_loopback_http() {
        for url in ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"] {
            assert!(validate_base_url(url).is_ok(), "loopback url {url} should be accepted");
        }
    }

    #[test]
    fn test_validate_base_url_rejects_non_loopback_http() {
        let err = validate_base_url("http://gateway.internal.example.com")
            .expect_err("non-loopback http must be rejected");
        assert!(err.is_validation(), "expected a validation error, got {err}");
        let message = err.to_string();
        assert!(message.contains("unencrypted"), "should explain the risk, got: {message}");
        assert!(message.contains("'http'"), "should name the scheme, got: {message}");
    }

    #[test]
    fn test_validate_base_url_rejects_other_schemes_and_garbage() {
        for url in ["ftp://files.example.com", "ws://gateway.example.com", "not-a-url"] {
            let err = validate_base_url(url).expect_err("{url} must be rejected");
            assert!(err.is_validation(), "expected a validation error for {url}, got {err}");
        }
    }
}
