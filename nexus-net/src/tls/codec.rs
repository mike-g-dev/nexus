use std::io::{self, Read, Write};

use rustls::ClientConnection;
use rustls::pki_types::ServerName;

use super::{TlsConfig, TlsError};

/// Sans-IO TLS codec. Decrypts inbound bytes, encrypts outbound bytes.
///
/// Wraps a rustls `ClientConnection` with an API shaped for nexus-net.
/// The codec is a pure state machine: callers drive IO and buffering;
/// the codec only transforms bytes.
///
/// # API at a glance
///
/// - **Inbound:** [`read_tls`](Self::read_tls) feeds buffered ciphertext
///   one packet step at a time; [`read_tls_from`](Self::read_tls_from)
///   drives a sync [`Read`] source directly;
///   [`read_and_process_tls`](Self::read_and_process_tls) loops over
///   bounded input.
/// - **Drain plaintext:** [`read_plaintext`](Self::read_plaintext) into
///   a slice; [`drain_plaintext_into`](Self::drain_plaintext_into) feeds
///   any [`ParserSink`](crate::ParserSink) (e.g. `FrameReader`) with
///   one fewer copy.
/// - **Outbound:** [`encrypt`](Self::encrypt) returns bytes accepted
///   (chunked); [`write_tls_to`](Self::write_tls_to) drains ciphertext
///   to a writer.
/// - **Shutdown:** [`send_close_notify`](Self::send_close_notify)
///   queues the alert; flush via `write_tls_to` before transport close.
pub struct TlsCodec {
    inner: ClientConnection,
}

impl TlsCodec {
    /// Create a new TLS codec for the given hostname.
    ///
    /// The hostname is used for SNI (Server Name Indication) and
    /// certificate verification.
    pub fn new(config: &TlsConfig, hostname: &str) -> Result<Self, TlsError> {
        let server_name = ServerName::try_from(hostname.to_owned())
            .map_err(|_| TlsError::InvalidHostname(hostname.to_owned()))?;

        let conn = ClientConnection::new(config.inner.clone(), server_name)?;

        Ok(Self { inner: conn })
    }

    // =========================================================================
    // Inbound (socket → TLS → FrameReader)
    // =========================================================================

    /// Advance the codec by a single TLS packet step: one read + one
    /// `process_new_packets` pair.
    ///
    /// Returns the number of ciphertext bytes consumed from `src`. The
    /// caller drains any plaintext between calls (via
    /// [`read_plaintext`](Self::read_plaintext) or
    /// [`drain_plaintext_into`](Self::drain_plaintext_into)) — feeding
    /// more ciphertext while plaintext is queued can overflow rustls's
    /// internal plaintext buffer. This is the canonical primitive for
    /// streaming app-data adapters (poll socket → step codec → drain
    /// plaintext → repeat).
    ///
    /// For bounded input that fits in rustls's plaintext queue
    /// (handshake bytes, in-memory tests), use the drain-loop helper
    /// [`read_and_process_tls`](Self::read_and_process_tls).
    ///
    /// # Returns
    ///
    /// `Ok(0)` if `src` is empty, or if rustls's deframer cannot
    /// progress on the input alone (matches `Read::read` idiom — the
    /// caller's loop is responsible for detecting stuck state).
    /// Otherwise `Ok(n)` where `n > 0` is bytes consumed (always
    /// `<= src.len()`; rustls's deframer caps each call at its
    /// internal `READ_SIZE`).
    ///
    /// # Errors
    ///
    /// Any rustls error from the read or process step (alerts,
    /// decryption failures, plaintext-buffer overflow, protocol
    /// violations).
    #[inline]
    pub fn read_tls(&mut self, src: &[u8]) -> Result<usize, TlsError> {
        if src.is_empty() {
            return Ok(0);
        }
        let mut cursor = io::Cursor::new(src);
        let consumed = self.inner.read_tls(&mut cursor)?;
        if consumed > 0 {
            self.inner.process_new_packets()?;
        }
        Ok(consumed)
    }

    /// Feed buffered TLS bytes through rustls in a loop until the
    /// entire slice is consumed.
    ///
    /// **Use only for bounded input** that fits in rustls's plaintext
    /// queue — in-memory tests, custom adapters that pre-buffer a
    /// known-bounded byte sequence. Do **not** use for streaming app
    /// data: large ciphertext slices fed without intervening plaintext
    /// drains overflow rustls's internal plaintext buffer
    /// (`received plaintext buffer full`). For streaming adapters,
    /// drive [`read_tls`](Self::read_tls) step-by-step yourself.
    ///
    /// **No production callers in this crate** — kept as a
    /// user-facing safety helper. The async `TlsInner::connect`
    /// (nexus-async-web) and sync `TlsStream::connect` drive their
    /// own loops over [`read_tls`](Self::read_tls) /
    /// [`read_tls_from`](Self::read_tls_from). External adapter
    /// authors who pre-buffer ciphertext can reach for this helper
    /// to avoid reimplementing the consume-loop.
    ///
    /// # Why this exists
    ///
    /// `rustls::Connection::read_tls` is not guaranteed to consume the
    /// full provided slice on a single call. The naive pattern
    /// `codec.read_tls(&buf)?` silently drops the unconsumed tail
    /// (issue #200 — a TLS handshake against a server that splits its
    /// response into multiple records inside one TCP segment fails
    /// because the unconsumed bytes vanish). This helper encodes the
    /// correct loop so naive callers don't reintroduce the bug.
    ///
    /// # Returns
    ///
    /// `Ok(src.len())` when the entire slice has been consumed.
    ///
    /// # Errors
    ///
    /// - `TlsError::Io(InvalidData)` if rustls's deframer can't make
    ///   progress (returned 0 bytes consumed) — malformed input.
    /// - Any rustls error from the underlying read/process steps.
    pub fn read_and_process_tls(&mut self, src: &[u8]) -> Result<usize, TlsError> {
        let mut consumed = 0;
        while consumed < src.len() {
            let n = self.read_tls(&src[consumed..])?;
            if n == 0 {
                return Err(TlsError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "TLS codec stopped before consuming buffered input \
                     (rustls deframer cannot make progress)",
                )));
            }
            consumed += n;
        }
        Ok(consumed)
    }

    /// Drive a sync [`Read`] source: read up to rustls's internal
    /// `READ_SIZE` from `src`, then process the records.
    ///
    /// Equivalent to one `read_tls` step but pulls bytes from a
    /// `Read` source instead of a buffer. Returns the bytes read from
    /// `src`, or 0 on EOF / no bytes available. The caller's loop
    /// handles the rest.
    pub fn read_tls_from<R: Read>(&mut self, src: &mut R) -> Result<usize, TlsError> {
        let n = self.inner.read_tls(src)?;
        if n > 0 {
            self.inner.process_new_packets()?;
        }
        Ok(n)
    }

    /// Drain decrypted plaintext into a [`ParserSink`](crate::ParserSink).
    ///
    /// Direct-feed path: uses `BufRead::fill_buf` to borrow rustls's
    /// internal plaintext queue and copy directly into `sink.spare()`,
    /// skipping the intermediate `&mut [u8]` that the
    /// [`read_plaintext`](Self::read_plaintext) shape requires. Returns
    /// the number of plaintext bytes delivered.
    ///
    /// Implements the zero-copy seam between rustls and parsers
    /// (`FrameReader` for WebSocket framing, `ResponseReader` for
    /// HTTP). Used by adapters' `WireStream::poll_fill_into` to fold
    /// plaintext draining into the same call that drives ciphertext
    /// reads.
    pub fn drain_plaintext_into<P: crate::ParserSink>(
        &mut self,
        sink: &mut P,
    ) -> Result<usize, TlsError> {
        let mut rd = self.inner.reader();
        let chunk = match std::io::BufRead::fill_buf(&mut rd) {
            Ok(chunk) => chunk,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(0),
            Err(e) => return Err(TlsError::Io(e)),
        };
        if chunk.is_empty() {
            return Ok(0);
        }
        let spare = sink.spare();
        let n = chunk.len().min(spare.len());
        if n == 0 {
            // Sink has no room; caller must drain the parser before
            // we can deliver more plaintext.
            return Ok(0);
        }
        spare[..n].copy_from_slice(&chunk[..n]);
        sink.filled(n);
        std::io::BufRead::consume(&mut rd, n);
        Ok(n)
    }

    /// Read decrypted plaintext into a buffer (sans-IO path).
    ///
    /// For users who want to feed bytes into FrameReader manually
    /// or use a different parser.
    #[inline]
    pub fn read_plaintext(&mut self, dst: &mut [u8]) -> Result<usize, TlsError> {
        match self.inner.reader().read(dst) {
            Ok(n) => Ok(n),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(0),
            Err(e) => Err(TlsError::Io(e)),
        }
    }

    // =========================================================================
    // Outbound (FrameWriter → TLS → socket)
    // =========================================================================

    /// Encrypt up to `plaintext.len()` bytes, returning the number of
    /// bytes actually accepted by rustls's outbound plaintext queue.
    ///
    /// Chunked semantics — the caller's `write_all` (or equivalent)
    /// handles re-driving on partial acceptance. This is the
    /// `AsyncWrite::poll_write` contract: surface backpressure as a
    /// partial count, not a hard error.
    ///
    /// # Returns
    ///
    /// `Ok(0)` if rustls's queue is full and cannot accept any bytes
    /// (caller should drain ciphertext to the socket and retry).
    /// Otherwise `Ok(n)` where `n > 0` is plaintext bytes queued for
    /// encryption. `n` may be less than `plaintext.len()`.
    ///
    /// # Errors
    ///
    /// Any rustls writer error other than `WriteZero` (which is
    /// translated to `Ok(0)` so callers treat queue-full as
    /// backpressure rather than a hard failure).
    #[inline]
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<usize, TlsError> {
        match self.inner.writer().write(plaintext) {
            Ok(n) => Ok(n),
            Err(e) if e.kind() == io::ErrorKind::WriteZero => Ok(0),
            Err(e) => Err(TlsError::Io(e)),
        }
    }

    /// Set rustls's outbound plaintext queue limit. `None` for
    /// unlimited (rustls accepts as much plaintext as memory allows;
    /// pair with a caller-side bound).
    ///
    /// Default is rustls's `DEFAULT_BUFFER_LIMIT = 64 KiB`. Trading
    /// workloads with small messages typically don't need to change
    /// this. Bulk-transfer workloads (large snapshots, file uploads
    /// over TLS) may benefit from raising it to reduce drain/refill
    /// cycles in [`encrypt`](Self::encrypt).
    pub fn set_buffer_limit(&mut self, limit: Option<usize>) {
        self.inner.set_buffer_limit(limit);
    }

    /// Queue a TLS `close_notify` alert.
    ///
    /// Subsequent calls to [`wants_write`](Self::wants_write) will
    /// return true until the alert ciphertext has been written via
    /// [`write_tls_to`](Self::write_tls_to).
    ///
    /// Idempotent: rustls tracks whether close_notify has been sent
    /// and no-ops on duplicate calls.
    ///
    /// Use in `AsyncWrite::poll_shutdown` (or equivalent) before
    /// closing the underlying transport. Without close_notify, the
    /// peer sees TCP FIN as a potential truncation and may error its
    /// read loop mid-stream.
    #[inline]
    pub fn send_close_notify(&mut self) {
        self.inner.send_close_notify();
    }

    /// Flush encrypted bytes to a socket.
    ///
    /// Returns the number of bytes written. Call in a loop or when
    /// [`wants_write`](Self::wants_write) returns true.
    pub fn write_tls_to<W: Write>(&mut self, dst: &mut W) -> io::Result<usize> {
        self.inner.write_tls(dst)
    }

    // =========================================================================
    // State
    // =========================================================================

    /// Whether the TLS handshake is still in progress.
    #[inline]
    pub fn is_handshaking(&self) -> bool {
        self.inner.is_handshaking()
    }

    /// Whether the codec has buffered TLS data to read.
    #[inline]
    pub fn wants_read(&self) -> bool {
        self.inner.wants_read()
    }

    /// Whether the codec has encrypted data to write.
    #[inline]
    pub fn wants_write(&self) -> bool {
        self.inner.wants_write()
    }
}

impl std::fmt::Debug for TlsCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsCodec")
            .field("handshaking", &self.inner.is_handshaking())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use super::*;

    // -------------------------------------------------------------------------
    // In-memory handshake scaffolding (lifted from examples/perf_tls.rs).
    // -------------------------------------------------------------------------

    fn generate_self_signed() -> (Vec<rustls::pki_types::CertificateDer<'static>>, Vec<u8>) {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("cert generation");
        (
            vec![rustls::pki_types::CertificateDer::from(
                cert.cert.der().to_vec(),
            )],
            cert.key_pair.serialize_der(),
        )
    }

    /// Generate an N-cert ECDSA-P256 chain whose serialized DER pushes
    /// the TLS 1.3 server's first handshake burst past rustls's
    /// `READ_SIZE = 4096` per-call deframer cap. ECDSA keygen is
    /// microseconds (vs RSA-4096's ~1.5s per key) so this stays cheap
    /// even at chain depth 10.
    ///
    /// Why a deep chain instead of one big RSA cert: chain depth scales
    /// the Certificate message linearly without paying for slow RSA
    /// keygen. 10 P-256 certs ≈ 5KB of cert bytes, comfortably over
    /// 4096. Each link is signed by its parent — a real CA-style chain.
    ///
    /// Returns `(chain_in_send_order, leaf_key_der)`. The chain is
    /// `[leaf, intermediate_n, ..., intermediate_1, root]` — the order
    /// rustls sends in the Certificate message.
    fn generate_oversize_ecdsa_chain() -> (Vec<rustls::pki_types::CertificateDer<'static>>, Vec<u8>)
    {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};

        const CHAIN_DEPTH: usize = 10;

        // Generate the root + intermediates + leaf. Each non-leaf is a
        // CA-flagged cert that signs the next link.
        let mut keys: Vec<KeyPair> = Vec::with_capacity(CHAIN_DEPTH);
        let mut certs: Vec<rcgen::Certificate> = Vec::with_capacity(CHAIN_DEPTH);

        // Root.
        let root_key = KeyPair::generate().expect("root key");
        let mut root_params = CertificateParams::new(Vec::<String>::new()).expect("root params");
        root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let root_cert = root_params.self_signed(&root_key).expect("root self-sign");
        keys.push(root_key);
        certs.push(root_cert);

        // Intermediates (CHAIN_DEPTH - 2 of them, all CA-flagged).
        for _ in 0..(CHAIN_DEPTH - 2) {
            let key = KeyPair::generate().expect("int key");
            let mut params = CertificateParams::new(Vec::<String>::new()).expect("int params");
            params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
            let parent_cert = certs.last().expect("parent");
            let parent_key = keys.last().expect("parent key");
            let cert = params
                .signed_by(&key, parent_cert, parent_key)
                .expect("int signed");
            keys.push(key);
            certs.push(cert);
        }

        // Leaf (signed by the deepest intermediate, SAN=localhost).
        let leaf_key = KeyPair::generate().expect("leaf key");
        let leaf_params =
            CertificateParams::new(vec!["localhost".to_string()]).expect("leaf params");
        let parent_cert = certs.last().expect("parent");
        let parent_key = keys.last().expect("parent key");
        let leaf_cert = leaf_params
            .signed_by(&leaf_key, parent_cert, parent_key)
            .expect("leaf signed");

        // Server sends [leaf, intermediates_descending, root] in the
        // Certificate message. We built `certs` as [root, int_1, ...,
        // int_n], so reverse + prepend leaf.
        let mut chain: Vec<rustls::pki_types::CertificateDer<'static>> =
            Vec::with_capacity(CHAIN_DEPTH);
        chain.push(rustls::pki_types::CertificateDer::from(
            leaf_cert.der().to_vec(),
        ));
        for cert in certs.iter().rev() {
            chain.push(rustls::pki_types::CertificateDer::from(cert.der().to_vec()));
        }

        (chain, leaf_key.serialize_der())
    }

    /// In-memory pipe for handshake bytes.
    struct MemPipe {
        buf: Vec<u8>,
    }

    impl MemPipe {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }

        fn write_to(&mut self, data: &[u8]) {
            self.buf.extend_from_slice(data);
        }

        fn read_from(&mut self, dst: &mut [u8]) -> usize {
            let n = dst.len().min(self.buf.len());
            dst[..n].copy_from_slice(&self.buf[..n]);
            self.buf.drain(..n);
            n
        }

        fn len(&self) -> usize {
            self.buf.len()
        }
    }

    /// Build the server side and capture its first multi-record handshake
    /// burst (ServerHello + EncryptedExtensions + Certificate + CertVerify +
    /// Finished under TLS 1.3 — several records pushed back-to-back). The
    /// returned `server_out` is the slice we feed to the client `TlsCodec`
    /// to exercise the partial-consumption surface.
    fn setup_and_capture_server_burst(
        cert_chain: Vec<rustls::pki_types::CertificateDer<'static>>,
        key_der: Vec<u8>,
    ) -> (TlsCodec, rustls::ServerConnection, Vec<u8>) {
        let key = rustls::pki_types::PrivateKeyDer::try_from(key_der).unwrap();
        let server_config = Arc::new(
            rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(cert_chain, key)
                .unwrap(),
        );
        let mut server = rustls::ServerConnection::new(server_config).unwrap();

        let client_config = TlsConfig::builder().danger_no_verify().build().unwrap();
        let mut client = TlsCodec::new(&client_config, "localhost").unwrap();

        let mut c2s = MemPipe::new();
        let mut s2c = MemPipe::new();

        // Client writes ClientHello.
        // Loop `while wants_write()` (mirroring the server side below)
        // for defense-in-depth — if a future rustls or cert config splits
        // the ClientHello across multiple write batches, a single
        // write_tls_to call would leave bytes pending in the codec.
        while client.wants_write() {
            let mut cursor = Cursor::new(Vec::new());
            client.write_tls_to(&mut cursor).unwrap();
            c2s.write_to(cursor.get_ref());
        }

        // Server consumes ClientHello.
        let mut tmp = vec![0u8; 16384];
        let n = c2s.read_from(&mut tmp);
        server
            .read_tls(&mut Cursor::new(&tmp[..n]))
            .expect("server reads ClientHello");
        server.process_new_packets().unwrap();

        // Server writes its multi-record burst.
        while server.wants_write() {
            let mut cursor = Cursor::new(Vec::new());
            server.write_tls(&mut cursor).unwrap();
            s2c.write_to(cursor.get_ref());
        }

        let mut server_out = vec![0u8; s2c.len()];
        let n = s2c.read_from(&mut server_out);
        assert!(n > 0, "server should have produced handshake bytes");
        server_out.truncate(n);

        (client, server, server_out)
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    /// Regression test for issue #200.
    ///
    /// Pre-fix: `read_tls(&buf)` may consume only part of `buf`. Calling
    /// code in nexus-async-web + nexus-net's tls/stream.rs ignored the
    /// returned consumed count, dropping the unconsumed tail and stalling
    /// the TLS handshake. Post-fix: `read_and_process_tls` loops until the
    /// entire slice is consumed.
    #[test]
    fn read_and_process_tls_consumes_full_slice() {
        let (chain, key) = generate_self_signed();
        let (mut client, _server, server_out) = setup_and_capture_server_burst(chain, key);

        let consumed = client
            .read_and_process_tls(&server_out)
            .expect("helper must consume the full slice");

        assert_eq!(
            consumed,
            server_out.len(),
            "helper must consume every byte (issue #200)"
        );
        assert!(
            client.wants_write(),
            "client should have produced its handshake response"
        );
    }

    /// Stricter exercise: feed the captured server bytes one byte per
    /// `read_and_process_tls` call. Catches a class of bugs where the
    /// helper itself drops bytes between calls or skips the
    /// `process_new_packets` step in some iterations.
    #[test]
    fn read_and_process_tls_byte_at_a_time() {
        let (chain, key) = generate_self_signed();
        let (mut client, _server, server_out) = setup_and_capture_server_burst(chain, key);

        for byte in &server_out {
            client
                .read_and_process_tls(std::slice::from_ref(byte))
                .expect("byte-at-a-time must succeed");
        }

        assert!(
            client.wants_write(),
            "client should have produced its handshake response \
             after byte-at-a-time consumption"
        );
    }

    /// **The actual end-to-end regression test for issue #200.**
    ///
    /// The other tests in this module either don't exercise the helper's
    /// multi-iteration loop (`read_and_process_tls_consumes_full_slice`
    /// uses a small burst that consumes in one inner iteration;
    /// `read_and_process_tls_byte_at_a_time` invokes the helper many times
    /// with 1-byte slices but each invocation has a 1-iteration loop),
    /// or test only rustls's contract without exercising our helper
    /// (`bare_read_tls_partially_consumes_large_slice`).
    ///
    /// This test uses a 10-cert ECDSA-P256 chain to push the server's
    /// first handshake burst past rustls's `READ_SIZE = 4096` per-call
    /// cap. Chain depth (not key size) provides the bytes — keeps
    /// keygen fast. The helper is fed the whole burst in ONE call; its
    /// internal loop must iterate multiple times to consume everything.
    /// This is exactly the shape birch hit against polymarket.
    #[test]
    fn read_and_process_tls_handles_oversize_burst() {
        let (chain, key) = generate_oversize_ecdsa_chain();
        let (mut client, _server, server_out) = setup_and_capture_server_burst(chain, key);

        // Confirm the test is actually exercising the partial-consumption
        // path. If this assertion fails, future contributors investigating
        // know the burst-size assumption broke (e.g., rustls raised
        // READ_SIZE, or the cert chain shrank). Bump the chain size or
        // the key size in `generate_oversize_ecdsa_chain` to restore.
        assert!(
            server_out.len() > 4096,
            "burst must exceed READ_SIZE to exercise multi-iteration loop, \
             got {} bytes — bump cert chain in generate_oversize_ecdsa_chain",
            server_out.len()
        );

        let consumed = client
            .read_and_process_tls(&server_out)
            .expect("helper must consume the full slice across multiple iterations");

        assert_eq!(
            consumed,
            server_out.len(),
            "helper must consume every byte across the multi-iteration loop \
             (issue #200 — the actual partial-consumption surface)"
        );
        assert!(
            client.wants_write(),
            "client should have produced its handshake response after \
             consuming the oversize burst"
        );
    }

    /// Drive an in-memory TLS 1.3 handshake to completion.
    /// Returns the connected client codec + server connection ready for
    /// app-data exchange. Used by `read_tls_step` tests that need a
    /// post-handshake codec.
    fn connected_pair() -> (TlsCodec, rustls::ServerConnection) {
        let (cert_chain, key_der) = generate_self_signed();
        let key = rustls::pki_types::PrivateKeyDer::try_from(key_der).unwrap();
        let server_config = Arc::new(
            rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(cert_chain, key)
                .unwrap(),
        );
        let mut server = rustls::ServerConnection::new(server_config).unwrap();

        let client_config = TlsConfig::builder().danger_no_verify().build().unwrap();
        let mut client = TlsCodec::new(&client_config, "localhost").unwrap();

        let mut c2s = Vec::new();
        let mut s2c = Vec::new();

        for _ in 0..64 {
            while client.wants_write() {
                client.write_tls_to(&mut c2s).unwrap();
            }
            if !c2s.is_empty() {
                server.read_tls(&mut Cursor::new(&c2s)).unwrap();
                server.process_new_packets().unwrap();
                c2s.clear();
            }
            while server.wants_write() {
                server.write_tls(&mut s2c).unwrap();
            }
            if !s2c.is_empty() {
                client.read_and_process_tls(&s2c).unwrap();
                s2c.clear();
            }
            if !client.is_handshaking() && !server.is_handshaking() {
                return (client, server);
            }
        }
        panic!("TLS handshake did not complete");
    }

    /// Encrypt `payload` from the server side and capture the resulting
    /// ciphertext.
    fn encrypt_server_payload(server: &mut rustls::ServerConnection, payload: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        server.writer().write_all(payload).unwrap();
        let mut ciphertext = Vec::new();
        while server.wants_write() {
            server.write_tls(&mut ciphertext).unwrap();
        }
        ciphertext
    }

    /// Empty input is a cheap no-op, not an error.
    #[test]
    fn read_tls_empty_input_returns_zero() {
        let client_config = TlsConfig::builder().danger_no_verify().build().unwrap();
        let mut client = TlsCodec::new(&client_config, "localhost").unwrap();

        let n = client.read_tls(&[]).expect("empty input must not error");
        assert_eq!(n, 0);
    }

    /// Happy path: feed a small ciphertext prefix, get a non-zero
    /// consumed count back, drain the resulting plaintext.
    #[test]
    fn read_tls_normal_step() {
        let (mut client, mut server) = connected_pair();
        let payload = b"hello, world";
        let ciphertext = encrypt_server_payload(&mut server, payload);
        assert!(!ciphertext.is_empty());

        let consumed = client
            .read_tls(&ciphertext)
            .expect("step must succeed on fresh ciphertext");
        assert!(consumed > 0, "must consume at least one byte");
        assert!(consumed <= ciphertext.len());

        let mut dst = vec![0u8; payload.len()];
        let n = client.read_plaintext(&mut dst).unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(&dst[..n], payload);
    }

    /// Documentation pin: when a caller feeds ciphertext via repeated
    /// [`read_tls`] calls without ever draining plaintext between
    /// steps, rustls's internal plaintext buffer eventually overflows
    /// and surfaces as `received plaintext buffer full`. This is the
    /// constraint that motivates the streaming pattern (alternate
    /// step + drain) and the existence of [`read_and_process_tls`]
    /// for the bounded-input case where overflow can't happen.
    #[test]
    fn read_tls_rejects_when_caller_does_not_drain() {
        let (mut client, mut server) = connected_pair();
        let payload = vec![b'x'; 64 * 1024];
        let ciphertext = encrypt_server_payload(&mut server, &payload);

        // Drive read_tls in a loop without ever calling read_plaintext.
        // Plaintext queues until rustls's cap is hit, then the next
        // process_new_packets surfaces the overflow.
        let mut consumed = 0;
        let error = loop {
            match client.read_tls(&ciphertext[consumed..]) {
                Ok(0) => panic!("unexpected stuck state; consumed={consumed}"),
                Ok(n) => consumed += n,
                Err(e) => break e,
            }
            assert!(
                consumed < ciphertext.len(),
                "expected error before consuming entire slice"
            );
        };
        assert!(
            error.to_string().contains("received plaintext buffer full"),
            "unexpected error: {error}"
        );
    }

    /// `encrypt` returns the partial accepted count when rustls's
    /// outbound plaintext queue can't hold the full input. Lower the
    /// queue limit explicitly so the test doesn't depend on rustls's
    /// internal default.
    #[test]
    fn encrypt_returns_partial_when_queue_fills() {
        let (mut client, _server) = connected_pair();
        client.set_buffer_limit(Some(4096));

        // First 4 KiB fits.
        let n1 = client.encrypt(&[b'a'; 4096]).unwrap();
        assert_eq!(n1, 4096);

        // Next chunk: queue is full. encrypt accepts 0.
        let n2 = client.encrypt(&[b'b'; 4096]).unwrap();
        assert_eq!(n2, 0, "queue full → encrypt must report 0 accepted");
    }

    /// `set_buffer_limit(None)` lifts the cap entirely — `encrypt`
    /// accepts everything in one shot.
    #[test]
    fn set_buffer_limit_none_unlimits_queue() {
        let (mut client, _server) = connected_pair();
        client.set_buffer_limit(None);

        // Heap-allocated to avoid a 256 KiB stack frame in this test.
        let payload = vec![b'x'; 256 * 1024];
        let n = client.encrypt(&payload).unwrap();
        assert_eq!(
            n,
            256 * 1024,
            "unlimited queue must accept the entire payload"
        );
    }

    /// `drain_plaintext_into` direct-feeds a [`ParserSink`] without
    /// the intermediate slice copy. Pins the zero-copy path that
    /// `WireStream::poll_fill_into` uses on TLS adapters.
    #[test]
    fn drain_plaintext_into_zero_copy_path() {
        struct CaptureSink {
            buf: Vec<u8>,
            committed: usize,
        }
        impl crate::ParserSink for CaptureSink {
            fn spare(&mut self) -> &mut [u8] {
                &mut self.buf[self.committed..]
            }
            fn filled(&mut self, n: usize) {
                self.committed += n;
            }
        }

        let (mut client, mut server) = connected_pair();
        let payload = b"hello-frames";
        let ciphertext = encrypt_server_payload(&mut server, payload);

        // Step the codec until plaintext is queued.
        let mut consumed = 0;
        while consumed < ciphertext.len() {
            consumed += client.read_tls(&ciphertext[consumed..]).unwrap();
        }

        let mut sink = CaptureSink {
            buf: vec![0u8; 64],
            committed: 0,
        };
        let n = client
            .drain_plaintext_into(&mut sink)
            .expect("drain_plaintext_into must succeed");
        assert_eq!(n, payload.len(), "must feed all queued plaintext");
        assert_eq!(&sink.buf[..n], payload);

        // Idempotent on empty queue.
        let n = client.drain_plaintext_into(&mut sink).unwrap();
        assert_eq!(n, 0, "no more plaintext → Ok(0)");
    }

    /// `read_tls_from` drives a sync [`Read`] source: reads up to one
    /// `READ_SIZE` from the source, processes packets, returns bytes
    /// pulled. Verifies the read+process pair fold (caller no longer
    /// has to call `process_new_packets` after).
    #[test]
    fn read_tls_from_drives_sync_read_source() {
        let (mut client, mut server) = connected_pair();
        let payload = b"hello-from-source";
        let ciphertext = encrypt_server_payload(&mut server, payload);

        let mut cursor = Cursor::new(ciphertext);
        let mut total = 0;
        while total < cursor.get_ref().len() {
            let n = client.read_tls_from(&mut cursor).unwrap();
            if n == 0 {
                break;
            }
            total += n;
        }
        assert!(total > 0);

        let mut dst = vec![0u8; payload.len()];
        let n = client.read_plaintext(&mut dst).unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(&dst[..n], payload);
    }
}
