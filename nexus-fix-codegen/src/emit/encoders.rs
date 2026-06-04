use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::dict::FieldType;

use super::{
    AccKind, HEADER, RField, RMember, RMessage, acc_kind, acc_return_type, byte_lit, pascal,
    screaming, snake,
};

/// Tags owned by FrameWriter — excluded from the header encoder typestate.
const FRAME_TAGS: &[u32] = &[8, 9, 10, 35];

pub fn emit(messages: &[RMessage], header_fields: &[RField]) -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    s.push_str("use core::marker::PhantomData;\n\n");

    let enc_fields = build_enc_fields(header_fields);
    emit_header_encoder(&mut s, &enc_fields);
    for m in messages {
        emit_encoder(&mut s, m);
    }
    s
}

// =============================================================================
// Venue-shared header encoder (ordered typestate)
// =============================================================================

/// How a header field is encoded from its typed value.
enum HStyle {
    Bytes,
    Bool,
    Char,
    Int,
    SeqNum,
    Uint,
    Timestamp,
    Encoded,
}

struct EncHeaderField {
    tag: u32,
    method: String,
    required: bool,
    style: HStyle,
    kind: AccKind,
}

fn build_enc_fields(header_fields: &[RField]) -> Vec<EncHeaderField> {
    header_fields
        .iter()
        .filter(|f| !FRAME_TAGS.contains(&f.number))
        .map(|f| {
            let style = match acc_kind(f.ftype) {
                AccKind::Bytes | AccKind::Text => HStyle::Bytes,
                AccKind::Bool => HStyle::Bool,
                AccKind::Char => HStyle::Char,
                AccKind::I64 => HStyle::Int,
                AccKind::U64 => HStyle::SeqNum,
                AccKind::U32 | AccKind::DayOfMonth => HStyle::Uint,
                AccKind::Timestamp => HStyle::Timestamp,
                AccKind::Decimal
                | AccKind::Date
                | AccKind::Time
                | AccKind::MonthYear
                | AccKind::TzTime
                | AccKind::TzTimestamp
                | AccKind::Tenor => HStyle::Encoded,
            };
            let kind = acc_kind(f.ftype);
            EncHeaderField {
                tag: f.number,
                method: snake(&f.name),
                required: f.required,
                style,
                kind,
            }
        })
        .collect()
}

fn emit_header_encoder(s: &mut String, fields: &[EncHeaderField]) {
    let n = fields.len();

    s.push_str(
        "/// Venue-shared ordered header encoder.\n\
         ///\n\
         /// Walks the standard header fields in canonical wire order: required\n\
         /// fields are forced (you cannot reach `finish()` without them), optional\n\
         /// fields may be set or skipped. Generic over the message body stage `B`\n\
         /// it returns to via [`finish`](HeaderEncoder::finish).\n",
    );
    for i in 0..=n {
        let _ = writeln!(s, "pub enum HeaderS{i} {{}}");
    }
    s.push('\n');
    s.push_str(
        "pub struct HeaderEncoder<'buf, S, B> {\n    \
         frame: nexus_fix_codec::FrameWriter<'buf>,\n    \
         _marker: PhantomData<(S, B)>,\n}\n\n",
    );

    // Constructor on the initial state.
    s.push_str(
        "impl<'buf, B> HeaderEncoder<'buf, HeaderS0, B> {\n    \
         #[inline]\n    \
         fn start(frame: nexus_fix_codec::FrameWriter<'buf>) -> Self {\n        \
         HeaderEncoder { frame, _marker: PhantomData }\n    }\n}\n\n",
    );

    // Per-state setter windows: from state `i`, you may set field `i` and any
    // following *optional* fields up to and including the next required field.
    for i in 0..n {
        let next_req = (i..n).find(|&j| fields[j].required);
        let window_end = next_req.map_or(n, |r| r + 1);
        let _ = writeln!(s, "impl<'buf, B> HeaderEncoder<'buf, HeaderS{i}, B> {{");
        for (j, f) in fields.iter().enumerate().take(window_end).skip(i) {
            emit_header_setter(s, f, j + 1);
        }
        s.push_str("}\n\n");
    }

    // `finish()` is available on any state where no required field remains.
    // This includes the final state SN, plus any earlier state where all
    // remaining fields are optional (e.g., trailing optional fields).
    let finish_impl = |s: &mut String, state: usize| {
        let _ = write!(
            s,
            "impl<'buf, B: nexus_fix_codec::FromFrame<'buf>> HeaderEncoder<'buf, HeaderS{state}, B> {{\n    \
             /// Finish the header and continue to the message body stage.\n    \
             #[inline]\n    \
             pub fn finish(self) -> B {{\n        \
             B::from_frame(self.frame)\n    }}\n}}\n\n"
        );
    };
    for i in 0..=n {
        let all_remaining_optional = fields[i..].iter().all(|f| !f.required);
        if all_remaining_optional {
            finish_impl(s, i);
        }
    }
}

fn emit_header_setter(s: &mut String, f: &EncHeaderField, next: usize) {
    let tag = f.tag;
    let method = &f.method;
    let (param, body) = match f.style {
        HStyle::Bytes => (
            "&[u8]".to_string(),
            format!("self.frame.field({tag}, value);"),
        ),
        HStyle::Bool => (
            "bool".to_string(),
            format!("self.frame.field({tag}, &[nexus_fix_codec::encode_fix_bool(value)]);"),
        ),
        HStyle::Char => (
            "nexus_fix_codec::AsciiChar".to_string(),
            format!("self.frame.field({tag}, &[nexus_fix_codec::encode_fix_char(value)]);"),
        ),
        HStyle::Int => (
            "i64".to_string(),
            format!(
                "let mut tmp = [0u8; 20];\n        \
                 let n = nexus_fix_codec::encode_fix_int(value, &mut tmp);\n        \
                 self.frame.field({tag}, &tmp[..n]);"
            ),
        ),
        HStyle::SeqNum => (
            "u64".to_string(),
            format!(
                "let mut tmp = [0u8; 20];\n        \
                 let n = nexus_fix_codec::encode_fix_seqnum(value, &mut tmp);\n        \
                 self.frame.field({tag}, &tmp[..n]);"
            ),
        ),
        HStyle::Uint => (
            "u32".to_string(),
            format!(
                "let mut tmp = [0u8; 10];\n        \
                 let n = nexus_fix_codec::encode_fix_uint(value, &mut tmp);\n        \
                 self.frame.field({tag}, &tmp[..n]);"
            ),
        ),
        HStyle::Timestamp => (
            "nexus_fix_codec::FixTimestamp".to_string(),
            format!(
                "let mut tmp = [0u8; 32];\n        \
                 let n = value.encode(&mut tmp);\n        \
                 self.frame.field({tag}, &tmp[..n]);"
            ),
        ),
        HStyle::Encoded => {
            let param = acc_return_type(f.kind);
            (
                param.to_string(),
                format!(
                    "let mut tmp = [0u8; 32];\n        \
                     let n = value.encode(&mut tmp);\n        \
                     self.frame.field({tag}, &tmp[..n]);"
                ),
            )
        }
    };
    let _ = write!(
        s,
        "    #[inline]\n    \
         pub fn {method}(mut self, value: {param}) -> HeaderEncoder<'buf, HeaderS{next}, B> {{\n        \
         {body}\n        \
         HeaderEncoder {{ frame: self.frame, _marker: PhantomData }}\n    }}\n\n"
    );
}

// =============================================================================
// Per-message encoder: `{Msg}Encoder` (start) + `{Msg}Body` (typed setters)
// =============================================================================

fn emit_encoder(s: &mut String, m: &RMessage) {
    let enc = format!("{}Encoder", pascal(&m.name));
    let body = format!("{}Body", pascal(&m.name));
    let msgtype = byte_lit(&m.msgtype);

    let fields: Vec<&RField> = m
        .members
        .iter()
        .filter_map(|mem| match mem {
            RMember::Field(f) => Some(f),
            RMember::Group(_) => None,
        })
        .collect();

    // Length→Data pairs: the length setter is folded into the data setter.
    let mut paired_len: HashMap<u32, &RField> = HashMap::new();
    let mut data_nums: HashSet<u32> = HashSet::new();
    for w in fields.windows(2) {
        if let [l, d] = w
            && l.ftype == FieldType::Length
            && d.ftype == FieldType::Data
        {
            paired_len.insert(l.number, d);
            data_nums.insert(d.number);
        }
    }

    // --- start stage: wrap / wrap_reserved / header_encoder ---
    let _ = write!(
        s,
        "pub struct {enc}<'buf> {{\n    frame: nexus_fix_codec::FrameWriter<'buf>,\n}}\n\n"
    );
    let _ = writeln!(s, "impl<'buf> {enc}<'buf> {{");
    let _ = write!(
        s,
        "    /// Begin encoding into `buf` (writes `8=`, reserves `9=`, writes `35=`).\n    \
         pub fn wrap(buf: &'buf mut [u8]) -> Self {{\n        \
         Self {{ frame: nexus_fix_codec::FrameWriter::new(buf, super::BEGIN_STRING, {msgtype}) }}\n    }}\n\n"
    );
    let _ = write!(
        s,
        "    /// As [`wrap`](Self::wrap) with an explicit `8=…9=…` prefix reservation.\n    \
         pub fn wrap_reserved(buf: &'buf mut [u8], reserved: usize) -> Self {{\n        \
         Self {{ frame: nexus_fix_codec::FrameWriter::with_reserved(buf, super::BEGIN_STRING, {msgtype}, reserved) }}\n    }}\n\n"
    );
    let _ = write!(
        s,
        "    /// Enter the header: returns the ordered header typestate.\n    \
         pub fn header_encoder(self) -> HeaderEncoder<'buf, HeaderS0, {body}<'buf>> {{\n        \
         HeaderEncoder::start(self.frame)\n    }}\n}}\n\n"
    );

    // --- body stage: FromFrame + typed setters + finish ---
    let _ = write!(
        s,
        "pub struct {body}<'buf> {{\n    frame: nexus_fix_codec::FrameWriter<'buf>,\n}}\n\n"
    );
    let _ = write!(
        s,
        "impl<'buf> nexus_fix_codec::FromFrame<'buf> for {body}<'buf> {{\n    \
         #[inline]\n    \
         fn from_frame(frame: nexus_fix_codec::FrameWriter<'buf>) -> Self {{\n        \
         Self {{ frame }}\n    }}\n}}\n\n"
    );
    let _ = writeln!(s, "impl<'buf> {body}<'buf> {{");

    let mut seen = HashSet::new();
    for f in &fields {
        if !seen.insert(f.number) || data_nums.contains(&f.number) {
            continue;
        }
        if let Some(d) = paired_len.get(&f.number) {
            emit_data_setter(s, f, d);
        } else {
            emit_body_setter(s, f);
        }
    }

    s.push_str(
        "    /// Finish the message: write `9=<canonical>` and the checksum,\n    \
         /// returning the framed message slice.\n    \
         pub fn finish(self) -> Result<&'buf [u8], nexus_fix_codec::EncodeError> {\n        \
         self.frame.finish()\n    }\n}\n\n",
    );
}

/// A typed body setter (plus a `_bytes` raw escape for the parsed kinds).
fn emit_body_setter(s: &mut String, f: &RField) {
    let name = snake(&f.name);
    let tag = format!("super::fields::TAG_{}", screaming(&f.name));

    if f.is_enum {
        let ty = pascal(&f.name);
        if f.single_char {
            let _ = write!(
                s,
                "    pub fn {name}(mut self, value: super::fields::{ty}) -> Self {{\n        \
                 self.frame.field({tag}, &[value.as_byte()]);\n        self\n    }}\n\n"
            );
        } else {
            let _ = write!(
                s,
                "    pub fn {name}(mut self, value: super::fields::{ty}<'_>) -> Self {{\n        \
                 self.frame.field({tag}, value.as_bytes());\n        self\n    }}\n\n"
            );
        }
        emit_bytes_setter(s, &format!("{name}_bytes"), &tag);
        return;
    }

    let kind = acc_kind(f.ftype);
    match kind {
        // Bytes-native: the value already *is* its wire form.
        AccKind::Bytes | AccKind::Text => emit_bytes_setter(s, &name, &tag),
        _ => {
            let param = acc_return_type(kind);
            let encode = body_encode(kind, &tag);
            let _ = write!(
                s,
                "    pub fn {name}(mut self, value: {param}) -> Self {{\n        \
                 {encode}\n        self\n    }}\n\n"
            );
            emit_bytes_setter(s, &format!("{name}_bytes"), &tag);
        }
    }
}

/// A raw `&[u8]` setter (the primary form for text/data, the escape hatch
/// otherwise).
fn emit_bytes_setter(s: &mut String, name: &str, tag: &str) {
    let _ = write!(
        s,
        "    pub fn {name}(mut self, value: &[u8]) -> Self {{\n        \
         self.frame.field({tag}, value);\n        self\n    }}\n\n"
    );
}

/// The encode statement for a typed body field (everything before the `self`).
fn body_encode(kind: AccKind, tag: &str) -> String {
    match kind {
        AccKind::Char => {
            format!("self.frame.field({tag}, &[nexus_fix_codec::encode_fix_char(value)]);")
        }
        AccKind::Bool => {
            format!("self.frame.field({tag}, &[nexus_fix_codec::encode_fix_bool(value)]);")
        }
        AccKind::I64 => scratch_encode(tag, "nexus_fix_codec::encode_fix_int(value, &mut tmp)"),
        AccKind::U32 => scratch_encode(tag, "nexus_fix_codec::encode_fix_uint(value, &mut tmp)"),
        AccKind::U64 => scratch_encode(tag, "nexus_fix_codec::encode_fix_seqnum(value, &mut tmp)"),
        AccKind::DayOfMonth => scratch_encode(
            tag,
            "nexus_fix_codec::encode_fix_uint(u32::from(value), &mut tmp)",
        ),
        // Owned value types with an inherent `encode(&mut [u8]) -> usize`.
        AccKind::Decimal
        | AccKind::Timestamp
        | AccKind::Date
        | AccKind::Time
        | AccKind::MonthYear
        | AccKind::TzTime
        | AccKind::TzTimestamp
        | AccKind::Tenor => scratch_encode(tag, "value.encode(&mut tmp)"),
        AccKind::Bytes | AccKind::Text => unreachable!("bytes/text handled before body_encode"),
    }
}

fn scratch_encode(tag: &str, call: &str) -> String {
    format!(
        "let mut tmp = [0u8; 32];\n        \
         let n = {call};\n        \
         self.frame.field({tag}, &tmp[..n]);"
    )
}

/// Length+Data pair: one setter writes the length then the data verbatim.
fn emit_data_setter(s: &mut String, len: &RField, data: &RField) {
    let name = snake(&data.name);
    let data_tag = format!("super::fields::TAG_{}", screaming(&data.name));
    let len_tag = format!("super::fields::TAG_{}", screaming(&len.name));
    let _ = write!(
        s,
        "    pub fn {name}(mut self, value: &[u8]) -> Self {{\n        \
         let mut tmp = [0u8; 20];\n        \
         let n = nexus_fix_codec::encode_fix_uint(value.len() as u32, &mut tmp);\n        \
         self.frame.field({len_tag}, &tmp[..n]);\n        \
         self.frame.field({data_tag}, value);\n        self\n    }}\n\n"
    );
}
