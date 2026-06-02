//! Sans-IO WebSocket reader/writer — shared by all async backends.
//!
//! `WsReader` and `WsWriter` are thin async wrappers around nexus-web's
//! `FrameReader` and `FrameWriter`. They are runtime-agnostic — any
//! `S: WireStream + Unpin` transport works.

use std::io;
use std::pin::Pin;

use nexus_net::buf::WriteBuf;
use nexus_net::{ParserSink, WireStream};
use nexus_web::ws::{CloseCode, Error as WsError, FrameReader, FrameWriter, Message};

// =============================================================================
// Async I/O helpers (poll_fn wrappers over WireStream)
// =============================================================================

pub(crate) async fn fill_async<W: WireStream + Unpin, P: ParserSink>(
    s: &mut W,
    sink: &mut P,
    max: usize,
) -> io::Result<usize> {
    std::future::poll_fn(|cx| Pin::new(&mut *s).poll_fill_into(cx, sink, max)).await
}

pub(crate) async fn write_all_async<W: WireStream + Unpin>(
    s: &mut W,
    mut buf: &[u8],
) -> io::Result<()> {
    while !buf.is_empty() {
        let n = std::future::poll_fn(|cx| Pin::new(&mut *s).poll_write(cx, buf)).await?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write returned 0"));
        }
        buf = &buf[n..];
    }
    Ok(())
}

// =============================================================================
// WsReader / WsWriter — decomposed sans-IO halves
// =============================================================================

/// Read half of a WebSocket connection.
///
/// Owns the [`FrameReader`] and parse state. Messages returned by
/// [`recv`](Self::recv) borrow from this reader's internal buffer,
/// independently of any [`WsWriter`] or transport state.
///
/// # Example
///
/// ```ignore
/// let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
///     .connect("ws://localhost:8080/ws")
///     .await?;
///
/// while let Some(msg) = reader.recv(&mut conn).await? {
///     match msg {
///         Message::Ping(data) => writer.send_pong(&mut conn, data).await?,
///         Message::Text(text) => {
///             // text borrows from reader — writer and conn are independent
///             let response = process(text);
///             writer.send_text(&mut conn, &response).await?;
///         }
///         _ => {}
///     }
/// }
/// ```
pub struct WsReader {
    pub(crate) reader: FrameReader,
    pub(crate) max_read_size: usize,
}

impl WsReader {
    /// Construct from raw nexus-web types.
    ///
    /// For custom handshakes or testing. Prefer using
    /// [`WsStreamBuilder`](super::WsStreamBuilder) for normal connections.
    pub fn from_raw_parts(reader: FrameReader, max_read_size: usize) -> Self {
        Self {
            reader,
            max_read_size: max_read_size.max(1),
        }
    }

    /// Receive the next message, using `conn` for transport I/O.
    ///
    /// The returned [`Message`] borrows from this reader's internal
    /// buffer. Because the reader and writer are independent types,
    /// you can hold the message while sending a response through a
    /// [`WsWriter`].
    pub async fn recv<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
    ) -> Result<Option<Message<'_>>, WsError> {
        loop {
            if self.reader.poll()? {
                return Ok(self.reader.next()?);
            }

            if self.reader.should_compact() {
                self.reader.compact();
            }
            if self.reader.spare().is_empty() {
                self.reader.compact();
                if self.reader.spare().is_empty() {
                    return Ok(None);
                }
            }

            let n = fill_async(conn, &mut self.reader, self.max_read_size).await?;
            if n == 0 {
                return Ok(None);
            }
        }
    }

    /// Access the underlying [`FrameReader`].
    pub fn frame_reader(&self) -> &FrameReader {
        &self.reader
    }

    /// Mutable access to the underlying [`FrameReader`].
    pub fn frame_reader_mut(&mut self) -> &mut FrameReader {
        &mut self.reader
    }

    /// Override max bytes read per recv call.
    pub fn set_max_read_size(&mut self, n: usize) {
        self.max_read_size = n.max(1);
    }
}

/// Write half of a WebSocket connection.
///
/// Owns the [`FrameWriter`] and [`WriteBuf`]. Encodes outgoing
/// frames and flushes them through a transport connection passed
/// to each send method.
pub struct WsWriter {
    pub(crate) writer: FrameWriter,
    pub(crate) write_buf: WriteBuf,
}

impl WsWriter {
    /// Construct from raw nexus-web types.
    ///
    /// For custom handshakes or testing. Prefer using
    /// [`WsStreamBuilder`](super::WsStreamBuilder) for normal connections.
    pub fn from_raw_parts(writer: FrameWriter, write_buf: WriteBuf) -> Self {
        Self { writer, write_buf }
    }

    /// Send a text message.
    pub async fn send_text<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
        text: &str,
    ) -> Result<(), WsError> {
        self.writer
            .encode_text_into(text.as_bytes(), &mut self.write_buf);
        write_all_async(conn, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a binary message.
    pub async fn send_binary<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
        data: &[u8],
    ) -> Result<(), WsError> {
        self.writer.encode_binary_into(data, &mut self.write_buf);
        write_all_async(conn, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a ping.
    pub async fn send_ping<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
        data: &[u8],
    ) -> Result<(), WsError> {
        self.writer
            .encode_ping_into(data, &mut self.write_buf)
            .map_err(WsError::Encode)?;
        write_all_async(conn, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a pong.
    pub async fn send_pong<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
        data: &[u8],
    ) -> Result<(), WsError> {
        self.writer
            .encode_pong_into(data, &mut self.write_buf)
            .map_err(WsError::Encode)?;
        write_all_async(conn, self.write_buf.data()).await?;
        Ok(())
    }

    /// Initiate close handshake.
    pub async fn close<S: WireStream + Unpin>(
        &mut self,
        conn: &mut S,
        code: CloseCode,
        reason: &str,
    ) -> Result<(), WsError> {
        if code == CloseCode::NoStatus {
            let mut dst = [0u8; 14];
            let n = self.writer.encode_empty_close(&mut dst);
            write_all_async(conn, &dst[..n]).await?;
        } else {
            self.writer
                .encode_close_into(code.as_u16(), reason.as_bytes(), &mut self.write_buf)
                .map_err(WsError::Encode)?;
            write_all_async(conn, self.write_buf.data()).await?;
        }
        Ok(())
    }

    /// Access the underlying [`FrameWriter`].
    pub fn frame_writer(&self) -> &FrameWriter {
        &self.writer
    }
}
