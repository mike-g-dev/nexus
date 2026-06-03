mod encoders;
mod fields;
mod groups;
mod messages;

use std::fmt::{self, Write};

use crate::dict::{Dictionary, FieldType, Member};

#[derive(Debug)]
pub enum EmitError {
    UnknownField(String),
    UnknownComponent(String),
    EmptyGroup(String),
    DataInGroup(String),
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownField(n) => {
                write!(f, "field '{n}' referenced but not defined in <fields>")
            }
            Self::UnknownComponent(n) => write!(f, "component '{n}' referenced but not defined"),
            Self::EmptyGroup(n) => write!(f, "group '{n}' has no members"),
            Self::DataInGroup(n) => {
                write!(
                    f,
                    "group '{n}' contains a DATA field; not supported in this version"
                )
            }
        }
    }
}

impl std::error::Error for EmitError {}

pub struct RField {
    pub name: String,
    pub number: u32,
    pub ftype: FieldType,
    pub required: bool,
    pub is_enum: bool,
    pub single_char: bool,
}

pub struct RGroup {
    pub name: String,
    pub number: u32,
    pub delimiter: u32,
    pub required: bool,
    pub members: Vec<RMember>,
}

pub enum RMember {
    Field(RField),
    Group(RGroup),
}

pub struct RMessage {
    pub name: String,
    pub msgtype: String,
    pub members: Vec<RMember>,
}

pub struct GeneratedFile {
    pub name: String,
    pub source: String,
}

pub fn generate(dict: &Dictionary) -> Result<Vec<GeneratedFile>, EmitError> {
    let messages = dict
        .messages
        .iter()
        .map(|m| {
            Ok(RMessage {
                name: m.name.clone(),
                msgtype: m.msgtype.clone(),
                members: resolve(dict, &m.members)?,
            })
        })
        .collect::<Result<Vec<_>, EmitError>>()?;

    for m in &messages {
        for mem in &m.members {
            if let RMember::Group(g) = mem {
                check_no_data(g)?;
            }
        }
    }

    Ok(vec![
        GeneratedFile {
            name: "fields.rs".to_string(),
            source: fields::emit(dict),
        },
        GeneratedFile {
            name: "messages.rs".to_string(),
            source: messages::emit(&messages),
        },
        GeneratedFile {
            name: "groups.rs".to_string(),
            source: groups::emit(&messages),
        },
        GeneratedFile {
            name: "encoders.rs".to_string(),
            source: encoders::emit(&messages),
        },
        GeneratedFile {
            name: "mod.rs".to_string(),
            source: emit_mod(&messages, &dict.major, &dict.minor),
        },
    ])
}

fn resolve(dict: &Dictionary, members: &[Member]) -> Result<Vec<RMember>, EmitError> {
    let mut out = Vec::new();
    for m in members {
        match m {
            Member::Field { name, required } => {
                let def = dict
                    .field(name)
                    .ok_or_else(|| EmitError::UnknownField(name.clone()))?;
                out.push(RMember::Field(RField {
                    name: name.clone(),
                    number: def.number,
                    ftype: def.ftype,
                    required: *required,
                    is_enum: def.is_enum(),
                    single_char: def.single_char(),
                }));
            }
            Member::Component { name, .. } => {
                let members = dict
                    .component(name)
                    .ok_or_else(|| EmitError::UnknownComponent(name.clone()))?;
                out.extend(resolve(dict, members)?);
            }
            Member::Group(g) => {
                let def = dict
                    .field(&g.name)
                    .ok_or_else(|| EmitError::UnknownField(g.name.clone()))?;
                let members = resolve(dict, &g.members)?;
                let delimiter =
                    first_field(&members).ok_or_else(|| EmitError::EmptyGroup(g.name.clone()))?;
                out.push(RMember::Group(RGroup {
                    name: g.name.clone(),
                    number: def.number,
                    delimiter,
                    required: g.required,
                    members,
                }));
            }
        }
    }
    Ok(out)
}

fn check_no_data(g: &RGroup) -> Result<(), EmitError> {
    for m in &g.members {
        match m {
            RMember::Field(f) if f.ftype == FieldType::Data => {
                return Err(EmitError::DataInGroup(g.name.clone()));
            }
            RMember::Group(inner) => check_no_data(inner)?,
            RMember::Field(_) => {}
        }
    }
    Ok(())
}

fn first_field(members: &[RMember]) -> Option<u32> {
    members
        .iter()
        .map(|m| match m {
            RMember::Field(f) => f.number,
            RMember::Group(g) => g.number,
        })
        .next()
}

pub fn group_type(prefix: &str, name: &str) -> String {
    format!("{prefix}{}", pascal(name))
}

pub fn subtree_tags(members: &[RMember], out: &mut Vec<u32>) {
    for m in members {
        match m {
            RMember::Field(f) => out.push(f.number),
            RMember::Group(g) => {
                out.push(g.number);
                subtree_tags(&g.members, out);
            }
        }
    }
}

pub fn tag_or(tags: &[u32]) -> String {
    let mut v: Vec<u32> = tags.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        if j > i {
            parts.push(format!("{}..={}", v[i], v[j]));
        } else {
            parts.push(v[i].to_string());
        }
        i = j + 1;
    }
    parts.join(" | ")
}

enum AccKind {
    Bytes,
    Ascii,
    I64,
    U32,
    Bool,
}

fn acc_kind(ft: FieldType) -> AccKind {
    match ft {
        FieldType::Data => AccKind::Bytes,
        FieldType::Length | FieldType::NumInGroup => AccKind::U32,
        FieldType::Int | FieldType::SeqNum => AccKind::I64,
        FieldType::Bool => AccKind::Bool,
        FieldType::Ascii => AccKind::Ascii,
    }
}

pub fn emit_value_accessor(s: &mut String, f: &RField) {
    let name = snake(&f.name);
    match acc_kind(f.ftype) {
        AccKind::Bytes => {
            let _ = write!(
                s,
                "    pub fn {name}(&self) -> Option<&'buf [u8]> {{\n        if self.{name}.is_present() {{ Some(self.{name}.slice(self.buf)) }} else {{ None }}\n    }}\n\n"
            );
        }
        AccKind::Ascii => {
            let _ = write!(
                s,
                "    pub fn {name}(&self) -> Option<&'buf nexus_fix_codec::AsciiTextStr> {{\n        if self.{name}.is_present() {{ nexus_fix_codec::AsciiTextStr::try_from_bytes(self.{name}.slice(self.buf)).ok() }} else {{ None }}\n    }}\n\n"
            );
        }
        AccKind::I64 => {
            let _ = write!(
                s,
                "    pub fn {name}(&self) -> Option<i64> {{\n        if self.{name}.is_present() {{ nexus_fix_codec::parse_fix_int(self.{name}.slice(self.buf)).ok() }} else {{ None }}\n    }}\n\n"
            );
        }
        AccKind::U32 => {
            let _ = write!(
                s,
                "    pub fn {name}(&self) -> Option<u32> {{\n        if self.{name}.is_present() {{ nexus_fix_codec::parse_fix_uint(self.{name}.slice(self.buf)).ok() }} else {{ None }}\n    }}\n\n"
            );
        }
        AccKind::Bool => {
            let _ = write!(
                s,
                "    pub fn {name}(&self) -> Option<bool> {{\n        if self.{name}.is_present() {{ nexus_fix_codec::parse_fix_bool(self.{name}.slice(self.buf)).ok() }} else {{ None }}\n    }}\n\n"
            );
        }
    }
    if f.is_enum {
        emit_enum_accessor(s, f, &name);
    }
}

fn emit_enum_accessor(s: &mut String, f: &RField, name: &str) {
    let ty = pascal(&f.name);
    if f.single_char {
        let _ = write!(
            s,
            "    pub fn {name}_enum(&self) -> Option<super::fields::{ty}> {{\n        self.{name}.slice(self.buf).first().map(|&b| super::fields::{ty}::from_byte(b))\n    }}\n\n"
        );
    } else {
        let _ = write!(
            s,
            "    pub fn {name}_enum(&self) -> Option<super::fields::{ty}<'buf>> {{\n        if self.{name}.is_present() {{ nexus_fix_codec::AsciiTextStr::try_from_bytes(self.{name}.slice(self.buf)).ok().map(super::fields::{ty}::from_bytes) }} else {{ None }}\n    }}\n\n"
        );
    }
}

pub fn emit_group_accessor(s: &mut String, name: &str, iter: &str) {
    let _ = write!(
        s,
        "    pub fn {name}(&self) -> super::groups::{iter}<'buf> {{\n        super::groups::{iter}::new(self.buf, self.{name})\n    }}\n\n"
    );
}

fn emit_mod(messages: &[RMessage], major: &str, minor: &str) -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    s.push_str("pub mod fields { include!(\"fields.rs\"); }\n");
    s.push_str("pub mod messages { include!(\"messages.rs\"); }\n");
    s.push_str("pub mod groups { include!(\"groups.rs\"); }\n");
    s.push_str("pub mod encoders { include!(\"encoders.rs\"); }\n\n");
    if !major.is_empty() && !minor.is_empty() {
        let _ = writeln!(
            s,
            "pub const BEGIN_STRING: &[u8] = {};\n",
            byte_lit(&format!("FIX.{major}.{minor}"))
        );
    }
    s.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum MsgType {\n");
    for m in messages {
        let _ = writeln!(s, "    {},", pascal(&m.name));
    }
    s.push_str("}\n\nimpl MsgType {\n");
    s.push_str("    pub fn from_bytes(b: &[u8]) -> Option<Self> {\n        match b {\n");
    for m in messages {
        let _ = writeln!(
            s,
            "            {} => Some(Self::{}),",
            byte_lit(&m.msgtype),
            pascal(&m.name)
        );
    }
    s.push_str("            _ => None,\n        }\n    }\n\n");
    s.push_str("    pub fn as_bytes(self) -> &'static [u8] {\n        match self {\n");
    for m in messages {
        let _ = writeln!(
            s,
            "            Self::{} => {},",
            pascal(&m.name),
            byte_lit(&m.msgtype)
        );
    }
    s.push_str("        }\n    }\n}\n");
    s
}

pub const HEADER: &str = "// @generated by nexus-fix-codegen. Do not edit.\n\n";

pub fn pascal(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mut out = String::new();
    let mut upper_next = true;
    for c in cleaned.chars() {
        if c == '_' {
            upper_next = true;
        } else if upper_next {
            out.push(c.to_ascii_uppercase());
            upper_next = false;
        } else {
            out.push(c);
        }
    }
    ensure_ident(out)
}

pub fn snake(s: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_ascii_alphanumeric() {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            continue;
        }
        if c.is_ascii_uppercase() {
            let prev = i.checked_sub(1).map(|j| chars[j]);
            let next = chars.get(i + 1).copied();
            let boundary = matches!(prev, Some(p) if p.is_ascii_lowercase() || p.is_ascii_digit())
                || matches!((prev, next), (Some(p), Some(n))
                    if p.is_ascii_uppercase() && n.is_ascii_lowercase());
            if boundary && !out.is_empty() && !out.ends_with('_') {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    raw_if_keyword(out)
}

pub fn screaming(s: &str) -> String {
    snake(s).to_ascii_uppercase()
}

fn ensure_ident(mut s: String) -> String {
    if s.is_empty() {
        s.push('X');
    } else if s.as_bytes()[0].is_ascii_digit() {
        s.insert(0, '_');
    }
    s
}

fn raw_if_keyword(s: String) -> String {
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
        "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
        "ref", "return", "self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "abstract", "become", "box", "do", "final",
        "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try",
    ];
    if KEYWORDS.contains(&s.as_str()) {
        format!("r#{s}")
    } else {
        ensure_ident(s)
    }
}

pub fn byte_lit(s: &str) -> String {
    let mut out = String::from("b\"");
    for b in s.bytes() {
        match b {
            b'"' | b'\\' => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(b as char),
            _ => {
                let _ = write!(out, "\\x{b:02x}");
            }
        }
    }
    out.push('"');
    out
}
