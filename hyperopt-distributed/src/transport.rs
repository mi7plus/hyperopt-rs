//! Transport abstraction so the JSON protocol runs unchanged over plain TCP or
//! a TLS-wrapped stream.
//!
//! The protocol only needs a bidirectional byte stream, so both sides hold a
//! `Box<dyn ReadWrite + Send>`. A plain [`TcpStream`](std::net::TcpStream)
//! satisfies it directly; with the `tls` feature a `rustls` stream does too.

use std::io::{Read, Write};

/// A bidirectional byte stream (anything that is both [`Read`] and [`Write`]).
pub trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

/// A boxed, thread-transferable connection stream.
pub type Stream = Box<dyn ReadWrite + Send>;

#[cfg(feature = "tls")]
pub(crate) mod tls {
    //! TLS config construction from raw DER bytes, kept behind the `tls`
    //! feature. Uses `rustls` with the `ring` crypto provider so no external
    //! toolchain (OpenSSL) is required.

    use std::io;
    use std::sync::Arc;

    use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
    use rustls::{ClientConfig, ClientConnection, ServerConfig, ServerConnection};

    fn provider() -> Arc<rustls::crypto::CryptoProvider> {
        Arc::new(rustls::crypto::ring::default_provider())
    }

    fn to_io<E: std::fmt::Display>(e: E) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, e.to_string())
    }

    /// Build a server config from a DER certificate chain and a DER private key.
    pub fn server_config(
        cert_chain_der: Vec<Vec<u8>>,
        private_key_der: Vec<u8>,
    ) -> io::Result<Arc<ServerConfig>> {
        let certs: Vec<CertificateDer<'static>> =
            cert_chain_der.into_iter().map(CertificateDer::from).collect();
        let key = PrivateKeyDer::try_from(private_key_der).map_err(to_io)?;
        let config = ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(to_io)?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(to_io)?;
        Ok(Arc::new(config))
    }

    /// Build a client config that trusts exactly the given DER root certificates
    /// (the natural fit for a self-signed coordinator on a trusted network).
    pub fn client_config(root_cert_der: Vec<Vec<u8>>) -> io::Result<Arc<ClientConfig>> {
        let mut roots = rustls::RootCertStore::empty();
        for der in root_cert_der {
            roots.add(CertificateDer::from(der)).map_err(to_io)?;
        }
        let config = ClientConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .map_err(to_io)?
            .with_root_certificates(roots)
            .with_no_client_auth();
        Ok(Arc::new(config))
    }

    /// Server-side handshake wrapper for one accepted TCP stream.
    pub fn accept(
        config: Arc<ServerConfig>,
        tcp: std::net::TcpStream,
    ) -> io::Result<rustls::StreamOwned<ServerConnection, std::net::TcpStream>> {
        let conn = ServerConnection::new(config).map_err(to_io)?;
        Ok(rustls::StreamOwned::new(conn, tcp))
    }

    /// Client-side handshake wrapper for one outgoing TCP stream.
    pub fn connect(
        config: Arc<ClientConfig>,
        server_name: &str,
        tcp: std::net::TcpStream,
    ) -> io::Result<rustls::StreamOwned<ClientConnection, std::net::TcpStream>> {
        let name = ServerName::try_from(server_name.to_string()).map_err(to_io)?;
        let conn = ClientConnection::new(config, name).map_err(to_io)?;
        Ok(rustls::StreamOwned::new(conn, tcp))
    }
}

/// Constant-time equality for shared-secret tokens, so a token check does not
/// leak the secret through a timing side-channel.
pub(crate) fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}
