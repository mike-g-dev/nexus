//! TLS stream wrapper — implements `Read + Write` over the sans-IO codec.
//!
//! Wraps a transport stream `S` and a [`TlsCodec`] into a single type
//! that transparently encrypts/decrypts. Sync only — async TLS lives
//! in `nexus-async-web::maybe_tls` (which drives the same sans-IO
//! codec at the poll level).

use std::io::{self, Read, Write};

use super::codec::TlsCodec;

/// A stream that transparently encrypts and decrypts via [`TlsCodec`].
///
/// Implements `Read` and `Write` by routing through the TLS codec.
/// The inner stream `S` carries raw ciphertext; callers see plaintext.
///
/// Construct via [`connect`](Self::connect) — the handshake is driven
/// to completion before the value is returned.
pub struct TlsStream<S> {
    stream: S,
    codec: TlsCodec,
}

impl<S> TlsStream<S> {
    /// Access the underlying transport stream.
    pub fn stream(&self) -> &S {
        &self.stream
    }

    /// Mutable access to the underlying transport stream.
    pub fn stream_mut(&mut self) -> &mut S {
        &mut self.stream
    }

    /// Access the TLS codec.
    pub fn codec(&self) -> &TlsCodec {
        &self.codec
    }

    /// Mutable access to the TLS codec.
    pub fn codec_mut(&mut self) -> &mut TlsCodec {
        &mut self.codec
    }

    /// Decompose into the inner stream and codec.
    pub fn into_parts(self) -> (S, TlsCodec) {
        (self.stream, self.codec)
    }

    /// Set rustls's outbound plaintext queue limit. Convenience
    /// pass-through to [`TlsCodec::set_buffer_limit`].
    ///
    /// Default is rustls's `DEFAULT_BUFFER_LIMIT = 64 KiB`. Bulk-
    /// transfer workloads (large snapshots, file uploads over TLS)
    /// may benefit from raising it. `None` for unlimited (caller
    /// is responsible for not encrypting more than memory allows).
    pub fn set_buffer_limit(&mut self, limit: Option<usize>) {
        self.codec.set_buffer_limit(limit);
    }
}

impl<S: Read + Write> TlsStream<S> {
    /// Wrap a transport stream and drive the TLS handshake to
    /// completion. Returns a stream ready for plaintext I/O.
    pub fn connect(stream: S, codec: TlsCodec) -> Result<Self, super::TlsError> {
        let mut s = Self { stream, codec };
        s.handshake()?;
        Ok(s)
    }

    /// Drive the TLS handshake to completion (blocking).
    fn handshake(&mut self) -> Result<(), super::TlsError> {
        while self.codec.is_handshaking() {
            while self.codec.wants_write() {
                self.codec.write_tls_to(&mut self.stream)?;
            }
            if self.codec.wants_read() {
                // read_tls_from drives one per-call read against the
                // Read trait and processes the resulting records.
                // Ok(0) means the peer closed mid-handshake — surface
                // explicitly so we don't loop forever with
                // is_handshaking() still true.
                let n = self.codec.read_tls_from(&mut self.stream)?;
                if n == 0 {
                    return Err(super::TlsError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed during TLS handshake",
                    )));
                }
            }
        }
        // Flush any remaining handshake data (client Finished, etc).
        while self.codec.wants_write() {
            self.codec.write_tls_to(&mut self.stream)?;
        }
        Ok(())
    }
}

impl<S: Read + Write> Read for TlsStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Try reading plaintext that's already buffered.
        let n = self.codec.read_plaintext(buf).map_err(tls_to_io)?;
        if n > 0 {
            return Ok(n);
        }

        // Need more ciphertext from the transport.
        // TLS may consume records without producing plaintext (session
        // tickets, key updates). Loop until we get plaintext or EOF.
        loop {
            let tls_n = self
                .codec
                .read_tls_from(&mut self.stream)
                .map_err(tls_to_io)?;
            if tls_n == 0 {
                return Ok(0); // EOF
            }
            let n = self.codec.read_plaintext(buf).map_err(tls_to_io)?;
            if n > 0 {
                return Ok(n);
            }
        }
    }
}

impl<S: Read + Write> Write for TlsStream<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Sync `Write` is all-or-nothing by trait contract, so loop
        // until rustls accepts everything. Typical sync writes are
        // well under rustls's plaintext queue cap (64 KiB by default),
        // so this loop is one iteration in practice.
        let mut written = 0;
        while written < buf.len() {
            let n = self.codec.encrypt(&buf[written..]).map_err(tls_to_io)?;
            if n == 0 {
                // rustls's plaintext queue is full. Drain it to the
                // socket — that produces ciphertext from the queued
                // plaintext, freeing space for more `encrypt` calls.
                // Without this, retrying would just hit the same wall.
                while self.codec.wants_write() {
                    self.codec.write_tls_to(&mut self.stream)?;
                }
                // Queue should now be drained; retry encrypt. If still
                // zero, the queue limit is genuinely smaller than the
                // remaining input — surface explicitly.
                let n2 = self.codec.encrypt(&buf[written..]).map_err(tls_to_io)?;
                if n2 == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "rustls plaintext queue limit is smaller than \
                         the remaining input — the buffer-limit may \
                         have been set too low (raise via \
                         TlsCodec::set_buffer_limit or \
                         TlsBufferCapacities::rustls_plaintext_limit), \
                         or chunk the write into smaller pieces",
                    ));
                }
                written += n2;
            } else {
                written += n;
            }
        }
        while self.codec.wants_write() {
            self.codec.write_tls_to(&mut self.stream)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        while self.codec.wants_write() {
            self.codec.write_tls_to(&mut self.stream)?;
        }
        self.stream.flush()
    }
}

fn tls_to_io(e: super::TlsError) -> io::Error {
    match e {
        super::TlsError::Io(io) => io,
        other => io::Error::other(other),
    }
}
