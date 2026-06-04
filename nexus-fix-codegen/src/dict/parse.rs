use std::fmt;

use quick_xml::events::attributes::AttrError;
use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use super::{Dictionary, EnumValue, FieldDef, FieldType, GroupDef, Member, MessageDef, MsgCat};

#[derive(Debug)]
pub enum ParseError {
    Xml(quick_xml::Error),
    Attr(AttrError),
    MissingAttr { element: String, attr: String },
    BadNumber(String),
    NoRoot,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(e) => write!(f, "XML error: {e}"),
            Self::Attr(e) => write!(f, "attribute error: {e}"),
            Self::MissingAttr { element, attr } => {
                write!(f, "<{element}> missing required attribute '{attr}'")
            }
            Self::BadNumber(s) => write!(f, "invalid field number '{s}'"),
            Self::NoRoot => write!(f, "missing <fix> root element"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<quick_xml::Error> for ParseError {
    fn from(e: quick_xml::Error) -> Self {
        Self::Xml(e)
    }
}

impl From<AttrError> for ParseError {
    fn from(e: AttrError) -> Self {
        Self::Attr(e)
    }
}

pub fn parse(xml: &str) -> Result<Dictionary, ParseError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut major = String::new();
    let mut minor = String::new();
    let mut header = Vec::new();
    let mut trailer = Vec::new();
    let mut fields = Vec::new();
    let mut components = Vec::new();
    let mut messages = Vec::new();
    let mut seen_root = false;

    loop {
        match reader.read_event()? {
            Event::Start(e) => match e.name().as_ref() {
                b"fix" | b"fixt" => {
                    seen_root = true;
                    major = opt_attr(&e, "major")?.unwrap_or_default();
                    minor = opt_attr(&e, "minor")?.unwrap_or_default();
                }
                b"fields" => fields = read_fields(&mut reader)?,
                b"components" => components = read_components(&mut reader)?,
                b"messages" => messages = read_messages(&mut reader)?,
                b"header" => header = read_members(&mut reader, b"header")?,
                b"trailer" => trailer = read_members(&mut reader, b"trailer")?,
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }

    if !seen_root {
        return Err(ParseError::NoRoot);
    }
    Ok(Dictionary {
        major,
        minor,
        header,
        trailer,
        fields,
        components,
        messages,
    })
}

fn read_fields(reader: &mut Reader<&[u8]>) -> Result<Vec<FieldDef>, ParseError> {
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Empty(e) if e.name().as_ref() == b"field" => {
                out.push(field_def(&e, Vec::new())?);
            }
            Event::Start(e) if e.name().as_ref() == b"field" => {
                let values = read_values(reader)?;
                out.push(field_def(&e, values)?);
            }
            Event::End(e) if e.name().as_ref() == b"fields" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn field_def(e: &BytesStart, values: Vec<EnumValue>) -> Result<FieldDef, ParseError> {
    let number_str = req_attr(e, "number")?;
    let number = number_str
        .parse::<u32>()
        .map_err(|_| ParseError::BadNumber(number_str))?;
    let ftype = opt_attr(e, "type")?.map_or(FieldType::String, |t| FieldType::from_str(&t));
    Ok(FieldDef {
        number,
        name: req_attr(e, "name")?,
        ftype,
        values,
    })
}

fn read_values(reader: &mut Reader<&[u8]>) -> Result<Vec<EnumValue>, ParseError> {
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"value" => {
                out.push(EnumValue {
                    value: req_attr(&e, "enum")?,
                    name: req_attr(&e, "description")?,
                });
            }
            Event::End(e) if e.name().as_ref() == b"field" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn read_components(reader: &mut Reader<&[u8]>) -> Result<Vec<(String, Vec<Member>)>, ParseError> {
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.name().as_ref() == b"component" => {
                let name = req_attr(&e, "name")?;
                let members = read_members(reader, b"component")?;
                out.push((name, members));
            }
            Event::End(e) if e.name().as_ref() == b"components" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn read_messages(reader: &mut Reader<&[u8]>) -> Result<Vec<MessageDef>, ParseError> {
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.name().as_ref() == b"message" => {
                let name = req_attr(&e, "name")?;
                let msgtype = req_attr(&e, "msgtype")?;
                let msgcat = match opt_attr(&e, "msgcat")?.as_deref() {
                    Some("admin") => MsgCat::Admin,
                    _ => MsgCat::App,
                };
                let members = read_members(reader, b"message")?;
                out.push(MessageDef {
                    name,
                    msgtype,
                    msgcat,
                    members,
                });
            }
            Event::End(e) if e.name().as_ref() == b"messages" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn read_members(reader: &mut Reader<&[u8]>, end: &[u8]) -> Result<Vec<Member>, ParseError> {
    let mut out = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Empty(e) => match e.name().as_ref() {
                b"field" => out.push(Member::Field {
                    name: req_attr(&e, "name")?,
                    required: is_required(&e)?,
                }),
                b"component" => out.push(Member::Component {
                    name: req_attr(&e, "name")?,
                }),
                _ => {}
            },
            Event::Start(e) => match e.name().as_ref() {
                b"group" => {
                    let name = req_attr(&e, "name")?;
                    let required = is_required(&e)?;
                    let members = read_members(reader, b"group")?;
                    out.push(Member::Group(GroupDef {
                        name,
                        required,
                        members,
                    }));
                }
                other => skip_to_end(reader, other)?,
            },
            Event::End(e) if e.name().as_ref() == end => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn skip_to_end(reader: &mut Reader<&[u8]>, end: &[u8]) -> Result<(), ParseError> {
    let mut depth = 1usize;
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.name().as_ref() == end => depth += 1,
            Event::End(e) if e.name().as_ref() == end => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(())
}

fn is_required(e: &BytesStart) -> Result<bool, ParseError> {
    Ok(opt_attr(e, "required")?.as_deref() == Some("Y"))
}

fn opt_attr(e: &BytesStart, key: &str) -> Result<Option<String>, ParseError> {
    match e.try_get_attribute(key)? {
        Some(a) => Ok(Some(a.unescape_value()?.into_owned())),
        None => Ok(None),
    }
}

fn req_attr(e: &BytesStart, key: &str) -> Result<String, ParseError> {
    opt_attr(e, key)?.ok_or_else(|| ParseError::MissingAttr {
        element: String::from_utf8_lossy(e.name().as_ref()).into_owned(),
        attr: key.to_string(),
    })
}
