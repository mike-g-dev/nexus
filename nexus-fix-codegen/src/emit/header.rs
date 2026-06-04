use std::fmt::Write;

use super::HEADER;

struct HeaderField {
    tag: u32,
    name: &'static str,
    ret_type: &'static str,
}

const HEADER_FIELDS: &[HeaderField] = &[
    HeaderField {
        tag: 8,
        name: "begin_string",
        ret_type: "&'buf [u8]",
    },
    HeaderField {
        tag: 9,
        name: "body_length",
        ret_type: "u32",
    },
    HeaderField {
        tag: 35,
        name: "msg_type",
        ret_type: "&'buf [u8]",
    },
    HeaderField {
        tag: 34,
        name: "msg_seq_num",
        ret_type: "u64",
    },
    HeaderField {
        tag: 49,
        name: "sender_comp_id",
        ret_type: "&'buf nexus_fix_codec::AsciiTextStr",
    },
    HeaderField {
        tag: 56,
        name: "target_comp_id",
        ret_type: "&'buf nexus_fix_codec::AsciiTextStr",
    },
    HeaderField {
        tag: 43,
        name: "poss_dup_flag",
        ret_type: "bool",
    },
    HeaderField {
        tag: 52,
        name: "sending_time",
        ret_type: "nexus_fix_codec::FixTimestamp",
    },
];

pub fn emit() -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    emit_struct(&mut s);
    emit_decode(&mut s);
    emit_accessors(&mut s);
    s
}

fn emit_struct(s: &mut String) {
    s.push_str("pub struct HeaderDecoder<'buf> {\n");
    s.push_str("    pub(super) reader: nexus_fix_codec::FieldReader<'buf>,\n");
    s.push_str("    pub(super) overflow: Option<nexus_fix_codec::RawField>,\n");
    for f in HEADER_FIELDS {
        let _ = writeln!(s, "    {}: nexus_fix_codec::FieldSpan,", f.name);
    }
    s.push_str("}\n\n");
}

fn emit_decode(s: &mut String) {
    s.push_str("impl<'buf> HeaderDecoder<'buf> {\n");
    s.push_str("    pub fn decode(buf: &'buf [u8]) -> Self {\n");
    s.push_str("        let mut h = Self {\n");
    s.push_str("            reader: nexus_fix_codec::FieldReader::new(buf, 0),\n");
    s.push_str("            overflow: None,\n");
    for f in HEADER_FIELDS {
        let _ = writeln!(
            s,
            "            {}: nexus_fix_codec::FieldSpan::EMPTY,",
            f.name
        );
    }
    s.push_str("        };\n");
    s.push_str("        while let Some(f) = h.reader.next_field() {\n");
    s.push_str("            match f.tag {\n");
    for f in HEADER_FIELDS {
        let _ = writeln!(s, "                {} => h.{} = f.value,", f.tag, f.name);
    }
    s.push_str("                _ => {\n");
    s.push_str("                    h.overflow = Some(f);\n");
    s.push_str("                    break;\n");
    s.push_str("                }\n");
    s.push_str("            }\n");
    s.push_str("        }\n");
    s.push_str("        h\n");
    s.push_str("    }\n\n");
}

fn emit_accessors(s: &mut String) {
    s.push_str("    pub fn buf(&self) -> &'buf [u8] { self.reader.buf() }\n\n");

    for f in HEADER_FIELDS {
        let _ = write!(
            s,
            "    pub fn {}(&self) -> Option<nexus_fix_codec::FieldView<'buf, {}>> {{\n        \
             nexus_fix_codec::FieldView::new(self.{}, self.reader.buf())\n    \
             }}\n\n",
            f.name, f.ret_type, f.name,
        );
    }

    s.push_str("    pub fn msg_type_enum(&self) -> Option<super::MsgType> {\n");
    s.push_str(
        "        if self.msg_type.is_present() { super::MsgType::from_bytes(self.msg_type.slice(self.reader.buf())) } else { None }\n",
    );
    s.push_str("    }\n");

    s.push_str("}\n");
}
