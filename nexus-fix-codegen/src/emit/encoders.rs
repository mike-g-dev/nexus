use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::dict::FieldType;

use super::{HEADER, RField, RMember, RMessage, pascal, screaming, snake};

pub fn emit(messages: &[RMessage]) -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    for m in messages {
        emit_encoder(&mut s, m);
    }
    s
}

fn emit_encoder(s: &mut String, m: &RMessage) {
    let ty = format!("{}Encoder", pascal(&m.name));
    let fields: Vec<&RField> = m
        .members
        .iter()
        .filter_map(|mem| match mem {
            RMember::Field(f) => Some(f),
            RMember::Group(_) => None,
        })
        .collect();

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

    let _ = write!(
        s,
        "pub struct {ty}<'buf> {{\n    w: nexus_fix_codec::FieldWriter<'buf>,\n}}\n\n"
    );
    let _ = writeln!(s, "impl<'buf> {ty}<'buf> {{");
    s.push_str(
        "    pub fn new(buf: &'buf mut [u8]) -> Self {\n        Self { w: nexus_fix_codec::FieldWriter::wrap(buf) }\n    }\n\n",
    );
    s.push_str(
        "    pub fn wrap_at(buf: &'buf mut [u8], offset: usize) -> Self {\n        Self { w: nexus_fix_codec::FieldWriter::wrap_at(buf, offset) }\n    }\n\n",
    );

    let mut seen = HashSet::new();
    for f in &fields {
        if !seen.insert(f.number) {
            continue;
        }
        if data_nums.contains(&f.number) {
            continue;
        }
        if let Some(d) = paired_len.get(&f.number) {
            emit_data_setter(s, f, d);
        } else {
            emit_field_setter(s, f);
        }
    }

    s.push_str("    pub fn finish(self) -> usize {\n        self.w.pos()\n    }\n\n");
    s.push_str(
        "    pub fn finish_with_checksum(mut self) -> usize {\n        let sum = nexus_fix_codec::checksum(self.w.data());\n        let cs = nexus_fix_codec::format_checksum(sum);\n        self.w.field(10, &cs);\n        self.w.pos()\n    }\n",
    );
    s.push_str("}\n\n");
}

fn emit_field_setter(s: &mut String, f: &RField) {
    let name = snake(&f.name);
    let tag = screaming(&f.name);
    let _ = write!(
        s,
        "    pub fn {name}(mut self, value: &[u8]) -> Self {{\n        self.w.field(super::fields::TAG_{tag}, value);\n        self\n    }}\n\n"
    );
    if f.is_enum {
        let ty = pascal(&f.name);
        if f.single_char {
            let _ = write!(
                s,
                "    pub fn {name}_value(mut self, value: super::fields::{ty}) -> Self {{\n        self.w.field(super::fields::TAG_{tag}, &[value.as_byte()]);\n        self\n    }}\n\n"
            );
        } else {
            let _ = write!(
                s,
                "    pub fn {name}_value(mut self, value: super::fields::{ty}<'_>) -> Self {{\n        self.w.field(super::fields::TAG_{tag}, value.as_bytes());\n        self\n    }}\n\n"
            );
        }
    }
}

fn emit_data_setter(s: &mut String, len: &RField, data: &RField) {
    let name = snake(&data.name);
    let data_tag = screaming(&data.name);
    let len_tag = screaming(&len.name);
    let _ = write!(
        s,
        "    pub fn {name}(mut self, value: &[u8]) -> Self {{\n        let mut tmp = [0u8; 20];\n        let n = nexus_fix_codec::encode_fix_seqnum(value.len() as u64, &mut tmp);\n        self.w.field(super::fields::TAG_{len_tag}, &tmp[..n]);\n        self.w.field(super::fields::TAG_{data_tag}, value);\n        self\n    }}\n\n"
    );
}
